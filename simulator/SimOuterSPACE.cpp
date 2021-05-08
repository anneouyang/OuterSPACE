#include "common.h"
#include "SimCycle.h"
#include "SimCache.h"

#include <Memory.h>

#include <cassert>
#include <iterator>
#include <algorithm>

#include <queue>
#include <deque>

typedef uintptr_t pointer_t;

struct OuterSPACEConfig {
    size_t NUM_PE = 256;
    size_t BLOCK_SIZE = 64;
    size_t DRAM_BANDWIDTH = 16 * 8 / 1.5;
};

static const OuterSPACEConfig config;

struct CSRRow {
    const CSRElement *data;
    index_t size;
};

struct MultiplyTask {
    CSRRow lmatCol, rmatRow;
    std::vector<CSRRow> result;
};

struct MergeTask {
    std::vector<CSRRow> inputs;
    CSRRow output;
};

class TaskProvider {
public:
    TaskProvider(const CSRMatrix &lmatCSC, const CSRMatrix &rmatCSR) : lmat(lmatCSC), rmat(rmatCSR) {
        assert(lmat.NRow() == rmat.NRow());

        index_t maxRowId = 0;
        for (size_t i = 0; i < lmat.data.size(); i++) {
            maxRowId = std::max(maxRowId, lmat.data[i].idx);
        }
        numRows = maxRowId + 1;

        multiplyPhase();
        mergePhase();
    }

    const std::vector<MultiplyTask> getMultiplyTasks() {
        return multTasks;
    }
    const std::vector<MergeTask> getMergeTasks() {
        return mergeTasks;
    }

private:
    CSRRow getRow(const CSRMatrix &mat, index_t rowId) {
        assert(rowId < mat.NRow());
        CSRRow result;
        result.data = &mat.data[0] + mat.pos[rowId];
        result.size = mat.pos[rowId + 1] - mat.pos[rowId];
        return result;
    }
    void multiplyPhase() {
        multResults.resize(numRows);

        for (index_t i = 0; i < rmat.NRow(); i++) {
            MultiplyTask task;
            task.lmatCol = getRow(lmat, i);
            task.rmatRow = getRow(rmat, i);

            if (task.lmatCol.size == 0 || task.rmatRow.size == 0)
                continue;

            for (index_t j = 0; j < task.lmatCol.size; j++) {
                index_t rowId = task.lmatCol.data[j].idx;
                std::vector<CSRElement> result;
                for (index_t k = 0; k < task.rmatRow.size; k++) {
                    result.push_back(CSRElement{ k, task.lmatCol.data[j].val * task.rmatRow.data[k].val });
                }
                multResults[rowId].push_back(std::move(result));
            }

            multTasks.push_back(task);
        }
    }
    void mergePhase() {
        mergedResult.pos.resize(numRows + 1);

        for (index_t i = 0; i < numRows; i++) {
            mergedResult.pos[i] = mergedResult.data.size();

            std::vector<CSRElement> buffer;
            for (auto &&v : multResults[i])
                std::copy(v.begin(), v.end(), std::back_inserter(buffer));
            std::sort(buffer.begin(), buffer.end(), [](const CSRElement &lhs, const CSRElement &rhs) { return lhs.idx < rhs.idx; });

            MergeTask task;
            for (auto &&v : multResults[i]) {
                CSRRow row;
                row.data = &v[0];
                row.size = v.size();
                task.inputs.push_back(row);
            }
            task.output.data = &mergedResult.data[0] + mergedResult.pos[i];
            task.output.size = 0;

            for (size_t j = 0; j < buffer.size(); j++) {
                if (j == 0 || buffer[j].idx == buffer[j - 1].idx) {
                    mergedResult.data.push_back(buffer[j]);
                    task.output.size++;
                } else {
                    mergedResult.data.back().val += buffer[j].val;
                }
            }

            mergeTasks.push_back(task);
        }

        mergedResult.pos[numRows] = mergedResult.data.size();
    }


private:
    const CSRMatrix &lmat, &rmat;
    index_t numRows;

    std::vector<std::vector<std::vector<CSRElement>>> multResults;  // Row ID -> # Merge Ways -> Elements
    CSRMatrix mergedResult;

    std::vector<MultiplyTask> multTasks;
    std::vector<MergeTask>    mergeTasks;
};

template<typename T>
struct TaskDispatcher {
    virtual bool haveTask(size_t peid) = 0;
    virtual T nextTask(size_t peid) = 0;
};

template<typename T>
class TaskDispatcherStatic : public TaskDispatcher<T> {
public:
    TaskDispatcherStatic(const std::vector<T> &tasks, size_t numPEs) : peTasks(numPEs) {
        size_t taskPerPE = (tasks.size() + numPEs - 1) / numPEs;

        for (size_t i = 0; i < tasks.size(); i++) {
            peTasks[i / taskPerPE].push_back(tasks[i]);
        }
    }

    virtual bool haveTask(size_t peid) override {
        return !peTasks[peid].empty();
    }
    virtual T nextTask(size_t peid) override {
        T task = std::move(peTasks[peid].front());
        peTasks[peid].pop_front();
        return task;
    }

private:
    std::vector<std::deque<T>> peTasks;
};

std::tuple<size_t, size_t> analyzeMultiplyTask(MultiplyTask task) {
    size_t workload = (size_t)task.lmatCol.size * task.rmatRow.size;
    size_t dramAccess = alignTo(workload * sizeof(CSRElement), config.BLOCK_SIZE) + alignTo(task.lmatCol.size * sizeof(CSRElement), config.BLOCK_SIZE) + alignTo(task.rmatRow.size * sizeof(CSRElement), config.BLOCK_SIZE);

    return { workload, dramAccess };
}

std::tuple<size_t, size_t> analyzeMergeTask(MergeTask task) {
    size_t workload = 0;
    for (auto &&in : task.inputs) {
        workload += in.size * task.inputs.size();
    }

    size_t dramAccess = 0;
    for (auto &&in : task.inputs) {
        dramAccess += alignTo(in.size * sizeof(CSRElement), config.BLOCK_SIZE);
    }
    dramAccess += alignTo(task.output.size * sizeof(CSRElement), config.BLOCK_SIZE);

    return { workload, dramAccess };
}

size_t analyzeCycles(std::tuple<size_t, size_t> taskParams) {
    auto [workload, dramAccess] = taskParams;
    size_t dramCycles = dramAccess * config.NUM_PE / config.DRAM_BANDWIDTH; // estimate the dram bandwidth per PE
    return std::max(workload, dramCycles);
}

size_t simulateOuterSPACEAnalytical(const CSRMatrix &lmatCSC, const CSRMatrix &rmatCSR) {
    TaskProvider provider(lmatCSC, rmatCSR);

    TaskDispatcherStatic<MultiplyTask> dispatcherMultiply(provider.getMultiplyTasks(), config.NUM_PE);
    TaskDispatcherStatic<MergeTask>    dispatcherMerge(provider.getMergeTasks(), config.NUM_PE);

    size_t maxCyclePE = 0;
    for (size_t i = 0; i < config.NUM_PE; i++) {
        size_t cycleCurrent = 0;
        while (dispatcherMultiply.haveTask(i)) {
            cycleCurrent += analyzeCycles(analyzeMultiplyTask(dispatcherMultiply.nextTask(i)));
        }
        maxCyclePE = std::max(maxCyclePE, cycleCurrent);
    }
    size_t cycleMultiply = maxCyclePE;

    maxCyclePE = 0;
    for (size_t i = 0; i < config.NUM_PE; i++) {
        size_t cycleCurrent = 0;
        while (dispatcherMerge.haveTask(i)) {
            cycleCurrent += analyzeCycles(analyzeMergeTask(dispatcherMerge.nextTask(i)));
        }
        maxCyclePE = std::max(maxCyclePE, cycleCurrent);
    }
    size_t cycleMerge = maxCyclePE;

    return cycleMultiply + cycleMerge;
}

struct MemoryRequest {
    pointer_t addr;
    bool is_write;
};
struct MemoryResponse {};

struct MemoryPortMaster {
    PortWrite<MemoryRequest> req;
    PortRead<MemoryResponse> resp;
};
struct MemoryPortSlave {
    PortRead<MemoryRequest> req;
    PortWrite<MemoryResponse> resp;
};

struct MemoryPortFIFO {
    static constexpr size_t DEFAULT_CAPACITY = 16;

    MemoryPortFIFO() : reqFIFO("MemoryReqFIFO", DEFAULT_CAPACITY, 1), respFIFO("MemoryReqFIFO", DEFAULT_CAPACITY, 1) {}
    MemoryPortFIFO(size_t reqCap, size_t respCap) : reqFIFO("MemoryReqFIFO", reqCap, 1), respFIFO("MemoryReqFIFO", respCap, 1) {}

    FIFO<MemoryRequest> reqFIFO;
    FIFO<MemoryResponse> respFIFO;

    MemoryPortMaster getMasterPort() {
        MemoryPortMaster port;
        port.req = reqFIFO.getWritePort();
        port.resp = respFIFO.getReadPort();
        return port;
    }
    MemoryPortSlave getSlavePort() {
        MemoryPortSlave port;
        port.req = reqFIFO.getReadPort();
        port.resp = respFIFO.getWritePort();
        return port;
    }
};

class Cache : public Module {
public:
    Cache(SimCache simulator) : simulator(std::move(simulator)), upstream(upstreamFIFO.getSlavePort()) {}

    MemoryPortMaster getUpstreamPort() { return upstreamFIFO.getMasterPort(); }
    void setDownstreamPort(MemoryPortMaster port) { this->downstream = port; }

    virtual void clockUpdate() override {
        if (downstream.resp.isReadable() && upstream.resp.isWritable()) {
            MemoryResponse resp = downstream.resp.read(1)[0];
            MemoryRequest  req  = pendingReqs.front();
            pendingReqs.pop_front();

            if (!req.is_write) {
                waitingRead = false;
            }
            upstream.resp.write({ resp });
        }

        if (upstream.req.isReadable() && downstream.req.isWritable()) {
            MemoryRequest req = upstream.req.peek(1)[0];
            bool consume = false;

            if (req.is_write) {
                consume = true;
                sendRequest(req);
            } else {
                if (!waitingRead) {
                    bool miss = simulator.access(req.addr);
                    if (miss) {
                        consume = true;
                        req.addr = simulator.getLineAddr(req.addr);
                        sendRequest(req);
                        waitingRead = true;
                    } else if (upstream.resp.isWritable()) {
                        consume = true;
                        upstream.resp.write({ MemoryResponse{} });
                    }
                }
            }

            if (consume) {
                upstream.req.read(1);
            }
        }
    }
    virtual void clockApply() override {}

private:
    void sendRequest(MemoryRequest req) {
        downstream.req.write({ req });
        pendingReqs.push_back(req);
    }

public:
    SimCache simulator;

private:
    MemoryPortFIFO upstreamFIFO;
    MemoryPortSlave upstream;
    MemoryPortMaster downstream;

    std::deque<MemoryRequest> pendingReqs;

    bool waitingRead = false;
};

class Crossbar : public Module {
public:
    Crossbar(size_t numUp, size_t numDown) : numUp(numUp), numDown(numDown), upstreamFIFOs(numUp), upstreams(numUp), downstreams(numDown), pendingReqs(numDown) {
        for (size_t i = 0; i < numUp; i++) {
            upstreams[i] = upstreamFIFOs[i].getSlavePort();
        }
    }

    void setMapper(std::function<size_t(pointer_t)> mapper) { this->mapper = mapper; }

    MemoryPortMaster getUpstreamPort(size_t idx) { return upstreamFIFOs[idx].getMasterPort(); }
    void setDownstreamPort(size_t idx, MemoryPortMaster port) { this->downstreams[idx] = port; }

    virtual void clockUpdate() override {
        processResp();
        processReq();
    }
    virtual void clockApply() override {}

private:
    void processResp() {
        std::vector<bool> busy(numUp, false);
        for (size_t i = 0; i < numDown; i++) {
            if (!downstreams[i].resp.isReadable())
                continue;
            MemoryResponse resp = downstreams[i].resp.peek(1)[0];
            size_t up = pendingReqs[i].front();
            if (busy[up] || !upstreams[up].resp.isWritable())
                continue;
            downstreams[i].resp.read(1);
            pendingReqs[i].pop_front();
            upstreams[up].resp.write({ resp });
            busy[up] = true;
        }
    }
    void processReq() {
        std::vector<bool> busy(numDown, false);
        for (size_t i = 0; i < numUp; i++) {
            if (!upstreams[i].req.isReadable())
                continue;
            MemoryRequest req = upstreams[i].req.peek(1)[0];
            size_t down = mapper(req.addr);
            if (busy[down] || !downstreams[down].req.isWritable())
                continue;
            upstreams[i].req.read(1);
            downstreams[down].req.write({ req });
            pendingReqs[down].push_back(i);
            busy[down] = true;
        }
    }

private:
    const size_t numUp, numDown;

    std::vector<MemoryPortFIFO> upstreamFIFOs;
    std::vector<MemoryPortSlave> upstreams;
    std::vector<MemoryPortMaster> downstreams;
    std::vector<std::deque<size_t>> pendingReqs;    // save the upstream port id

    std::function<size_t(pointer_t)> mapper;
};

class PE : public Module {
protected:
    using callback_t = std::function<void()>;

public:
    void setMemoryPort(MemoryPortMaster port) { this->port = port; }

    virtual void clockUpdate() override {
        if (port.resp.isReadable()) {
            port.resp.read(1);
            pendingReqs.front()();
            pendingReqs.pop_front();
        }

        if (port.req.isWritable()) {
            MemoryRequest req;
            callback_t callback;
            bool reqValid = true;

            if (!writeQueue.empty()) {
                req.addr = writeQueue.front().first;
                callback = writeQueue.front().second;
                req.is_write = true;
            } else if (!readQueue.empty()) {
                req.addr = readQueue.front().first;
                callback = readQueue.front().second;
                req.is_write = false;
            } else {
                reqValid = false;
            }

            if (reqValid) {
                port.req.write({ req });
                pendingReqs.push_back(callback);
            }
        }
        
        if (computing) {
            remainingCycles--;
            if (remainingCycles <= 0) {
                computeQueue.front().second();
                computeQueue.pop_front();

                computing = false;
            }
        }

        if (!computing) {
            computing = true;
            remainingCycles = computeQueue.front().first;
        }
    }

    virtual void clockApply() override {}

protected:
    std::deque<std::pair<pointer_t, callback_t>> readQueue;
    std::deque<std::pair<pointer_t, callback_t>> writeQueue;
    std::deque<std::pair<int, callback_t>>    computeQueue;

    bool computing = false;
    int remainingCycles = 0;

    MemoryPortMaster port;
    std::deque<callback_t> pendingReqs;
};

class PEMultiplier : public PE {
public:
    PEMultiplier(TaskDispatcher<MultiplyTask> &dispatcher, size_t peid) : dispatcher(dispatcher), peid(peid) {}

    virtual void clockUpdate() override {
        if (readQueue.empty() && dispatcher.haveTask(peid)) {
            MultiplyTask task = dispatcher.nextTask(peid);
            if (task.lmatCol.size != 0 && task.rmatRow.size != 0) {
                for (size_t i = 0; i < task.lmatCol.size; i++) {
                    readQueue.push_back({ pointer_t(task.lmatCol.data + i), callback_t([]() {}) });

                    pointer_t writeAddr = (pointer_t)&task.result[0];
                    size_t writeBytes = 0;
                    for (size_t j = 0; j < task.rmatRow.size; j++) {
                        bool needWrite = false;
                        writeBytes += sizeof(CSRElement);
                        if (writeBytes >= 64 || j + 1 >= task.rmatRow.size) {
                            needWrite = true;
                            writeAddr += 64;
                        }
                        readQueue.push_back({ pointer_t(task.rmatRow.data + j), callback_t([this, needWrite, writeAddr]() {
                            computeQueue.push_back({1, callback_t([this, needWrite, writeAddr]() {
                                if (needWrite)
                                    writeQueue.push_back({writeAddr, callback_t([]() {})});
                            })});
                        })});
                    }
                }
            }
        }

        PE::clockUpdate();
    }

private:
    TaskDispatcher<MultiplyTask> &dispatcher;
    size_t peid;
};

class DRAMBackend : public Module {
private:
    using Tech = ramulator::HBM;
    using Memory = ramulator::Memory<Tech, ramulator::Controller>;

    static constexpr size_t NUM_OUTSTANDING_REQUESTS = 128;
    static constexpr double DRAM_CLOCK_PERIOD = 1.5;
    static constexpr size_t DRAM_GRANULARITY = 32;

public:
    DRAMBackend(size_t channelWidth) : channelWidth(channelWidth), memFIFO(MemoryPortFIFO::DEFAULT_CAPACITY, ~0), memport(memFIFO.getSlavePort()) {
        ramulator::Config &configs = getConfig();

        // NOTE: spec and ctrls will be deleted by `memory', no unique_ptr required
        Tech *spec = new Tech(configs["org"], configs["speed"]);
        int C = configs.get_channels(), R = configs.get_ranks();
        // Check and Set channel, rank number
        spec->set_channel_number(C);
        spec->set_rank_number(R);
        std::vector<ramulator::Controller<Tech> *> ctrls;
        // fprintf(stderr, "DRAM C=%d\n", C);
        for (int c = 0; c < C; c++) {
            ramulator::DRAM<Tech> *channel = new ramulator::DRAM<Tech>(spec, Tech::Level::Channel);
            channel->id = c;
            channel->regStats("");
            ramulator::Controller<Tech> *ctrl = new ramulator::Controller<Tech>(configs, channel);
            ctrls.push_back(ctrl);
        }
        memory.reset(new Memory(configs, ctrls));
        // memory->type = Memory::Type::ChRaBaRoCo;
    }

    void setAddressTranslator(std::function<long(pointer_t)> func) { this->addrTranslator = func; }
    MemoryPortMaster getPort() { return memFIFO.getMasterPort(); }

    virtual void clockUpdate() override {
        if (memport.req.isReadable() && cntPendingReqs < NUM_OUTSTANDING_REQUESTS) {
            MemoryRequest req = memport.req.read(1)[0];
            long translatedAddr = addrTranslator(req.addr);

            for (size_t offset = 0; offset < channelWidth; offset += DRAM_GRANULARITY) {
                bool isLast = offset + DRAM_GRANULARITY >= channelWidth;
                ramulator::Request ramReq(translatedAddr + offset, req.is_write ? ramulator::Request::Type::WRITE : ramulator::Request::Type::READ, [this, isLast](ramulator::Request &r) {
                    if (isLast)
                        memport.resp.write({ MemoryResponse{} });
                    cntPendingReqs--;
                });
                cntPendingReqs++;
                requests.push(std::move(ramReq));
            }
        }

        if (cntClock >= DRAM_CLOCK_PERIOD) {
            cntClock -= DRAM_CLOCK_PERIOD;

            if (!requests.empty()) {
                memory->send(requests.front());
                requests.pop();
            }

            memory->tick();
        }
        cntClock += 1;
    }
    virtual void clockApply() override {}


private:
    static ramulator::Config &getConfig() {
        static const std::string strConfigFile = "HBM-Config.cfg";
        static bool inited = false;
        static ramulator::Config configs(strConfigFile);
        if (!inited) {
            configs.set_core_num(1);
            inited = true;
        }
        return configs;
    }

private:
    size_t channelWidth; // in bytes
    double cntClock = 0;
    size_t cntPendingReqs = 0;

    std::unique_ptr<Memory> memory;

    MemoryPortFIFO  memFIFO;
    MemoryPortSlave memport;

    std::function<long(pointer_t)> addrTranslator;

    std::queue<ramulator::Request> requests;
};