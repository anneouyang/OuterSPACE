#include "common.h"
#include "SimCycle.h"
#include "SimCache.h"

#include <Memory.h>

#include <cassert>
#include <memory>
#include <iterator>
#include <algorithm>

#include <queue>
#include <deque>

typedef uintptr_t pointer_t;

struct OuterSPACEConfig {
    size_t NUM_PE = 256;
    size_t NUM_PE_TILES = 16;
    size_t BLOCK_SIZE = 64;
    size_t L0_CACHE_NUM_GROUPS = 4;
    size_t L0_CACHE_SIZE = 16 * 1024;
    size_t NUM_DRAM_CHANNELS = 16;
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

    const std::vector<MultiplyTask> &getMultiplyTasks() {
        return multTasks;
    }
    const std::vector<MergeTask> &getMergeTasks() {
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
                task.result.push_back(CSRRow{&multResults[rowId].back()[0], multResults[rowId].back().size()});
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
        // size_t taskPerPE = (tasks.size() + numPEs - 1) / numPEs;

        for (size_t i = 0; i < tasks.size(); i++) {
            peTasks[i % numPEs].push_back(tasks[i]);
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

size_t simulateOuterSPACEAnalyticalMultiply(TaskProvider &provider) {
    TaskDispatcherStatic<MultiplyTask> dispatcherMultiply(provider.getMultiplyTasks(), config.NUM_PE);

    size_t maxCyclePE = 0;
    for (size_t i = 0; i < config.NUM_PE; i++) {
        size_t cycleCurrent = 0;
        while (dispatcherMultiply.haveTask(i)) {
            cycleCurrent += analyzeCycles(analyzeMultiplyTask(dispatcherMultiply.nextTask(i)));
        }
        maxCyclePE = std::max(maxCyclePE, cycleCurrent);
    }

    return maxCyclePE;
}

size_t simulateOuterSPACEAnalyticalMerge(TaskProvider &provider) {
    TaskDispatcherStatic<MergeTask>    dispatcherMerge(provider.getMergeTasks(), config.NUM_PE);

    size_t maxCyclePE = 0;
    for (size_t i = 0; i < config.NUM_PE; i++) {
        size_t cycleCurrent = 0;
        while (dispatcherMerge.haveTask(i)) {
            cycleCurrent += analyzeCycles(analyzeMergeTask(dispatcherMerge.nextTask(i)));
        }
        maxCyclePE = std::max(maxCyclePE, cycleCurrent);
    }

    return maxCyclePE;
}

size_t simulateOuterSPACEAnalytical(const CSRMatrix &lmatCSC, const CSRMatrix &rmatCSR) {
    TaskProvider provider(lmatCSC, rmatCSR);

    return simulateOuterSPACEAnalyticalMultiply(provider) + simulateOuterSPACEAnalyticalMerge(provider);
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
    Cache(SimCache simulator, std::string name = "") : Module(), simulator(std::move(simulator)), upstream(upstreamFIFO.getSlavePort()), name(name) {}

    MemoryPortMaster getUpstreamPort() { return upstreamFIFO.getMasterPort(); }
    void setDownstreamPort(MemoryPortMaster port) { this->downstream = port; }

    virtual void clockUpdate() override {
        bool respBusy = false;

        if (downstream.resp.isReadable() && upstream.resp.isWritable()) {
            MemoryResponse resp = downstream.resp.read(1)[0];
            MemoryRequest  req  = pendingReqs.front();
            pendingReqs.pop_front();

            if (!req.is_write) {
                waitingRead = false;
            }
            upstream.resp.write({ resp });

            respBusy = true;
        }

        if (upstream.req.isReadable() && downstream.req.isWritable()) {
            MemoryRequest req = upstream.req.peek(1)[0];
            bool consume = false;

            if (req.is_write) {
                consume = true;
                sendRequest(req);
                cntWrite++;
            } else {
                if (!waitingRead) {
                    bool miss = simulator.access(req.addr);
                    if (miss) {
                        consume = true;
                        req.addr = simulator.getLineAddr(req.addr);
                        sendRequest(req);
                        waitingRead = true;

                        cntRead++, cntReadMiss++;
                    } else if (upstream.resp.isWritable() && !respBusy && pendingReqs.empty()) {
                        consume = true;
                        upstream.resp.write({ MemoryResponse{} });

                        cntRead++;
                    }
                }
            }

            if (consume) {
                upstream.req.read(1);
            }
        }
    }
    virtual void clockApply() override {}

    virtual void printStats() {
        fprintf(stderr, "CACHE %s: Write %zu Read %zu Miss %zu (Miss Rate = %.3lf) Pending %zu\n", name.c_str(), cntWrite, cntRead, cntReadMiss, double(cntReadMiss) / cntRead, pendingReqs.size());
    }

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

    std::string name;
    size_t cntRead = 0, cntWrite = 0, cntReadMiss = 0;
};

class Crossbar : public Module {
public:
    Crossbar(size_t numUp, size_t numDown, std::string name) : numUp(numUp), numDown(numDown), upstreamFIFOs(numUp), upstreams(numUp), downstreams(numDown), pendingReqs(numDown), name(name) {
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

    virtual void printStats() {
        fprintf(stderr, "CROSSBAR %s: PENDING ", name.c_str());
        for (auto &&pending : pendingReqs)
            fprintf(stderr, "%zu ", pending.size());
        fprintf(stderr, "\n");
    }

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

    std::string name;
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
                writeQueue.pop_front();
            } else if (!readQueue.empty()) {
                req.addr = readQueue.front().first;
                callback = readQueue.front().second;
                req.is_write = false;
                readQueue.pop_front();
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

        if (!computing && !computeQueue.empty()) {
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
            dispatchedTasks++;
            if (task.lmatCol.size != 0 && task.rmatRow.size != 0) {
                for (size_t i = 0; i < task.lmatCol.size; i++) {
                    readQueue.push_back({ pointer_t(task.lmatCol.data + i), callback_t([]() {}) });

                    pointer_t writeAddr = (pointer_t)task.result[i].data;
                    size_t writeBytes = 0;
                    for (size_t j = 0; j < task.rmatRow.size; j++) {
                        bool needWrite = false;
                        writeBytes += sizeof(CSRElement);
                        if (writeBytes >= config.BLOCK_SIZE || j + 1 >= task.rmatRow.size) {
                            needWrite = true;
                            writeAddr += config.BLOCK_SIZE;
                            writeBytes -= config.BLOCK_SIZE;
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

    bool done() {
        return !dispatcher.haveTask(peid) && readQueue.empty() && computeQueue.empty() && writeQueue.empty();
    }

    virtual void printStats() {
        // return;
        fprintf(stderr, "PE %zu: Dispatched %zu tasks\n", peid, dispatchedTasks);
        fprintf(stderr, "PE %zu: Queue R %zu C %zu W %zu\n", peid, readQueue.size(), computeQueue.size(), writeQueue.size());
    }

private:
    TaskDispatcher<MultiplyTask> &dispatcher;
    size_t peid;

    size_t dispatchedTasks = 0;
};

class PEMerger : public PE {
public:
    PEMerger(TaskDispatcher<MergeTask> &dispatcher, size_t peid) : dispatcher(dispatcher), peid(peid) {}

    virtual void clockUpdate() override {
        if (readQueue.empty() && dispatcher.haveTask(peid)) {
            MergeTask task = dispatcher.nextTask(peid);
            dispatchedTasks++;
            auto [workload, dramAccess] = analyzeMergeTask(task);

            size_t writeBytes = task.output.size * sizeof(CSRElement);
            size_t numWrite = (writeBytes + config.BLOCK_SIZE - 1) / config.BLOCK_SIZE;
            pointer_t writeAddr = (pointer_t)task.output.data;

            for (size_t idx = 0; idx < task.inputs.size(); idx++) {
                auto &&input = task.inputs[idx];

                size_t readBytes = input.size * sizeof(CSRElement);
                size_t numRead = (readBytes + config.BLOCK_SIZE - 1) / config.BLOCK_SIZE;
                for (size_t i = 0; i < numRead; i++) {
                    bool isLast = idx + 1 >= task.inputs.size() && i + 1 >= numRead;
                    readQueue.push_back({ pointer_t(input.data + i * config.BLOCK_SIZE), callback_t([this, isLast, workload, numWrite, writeAddr]() {
                        if (isLast) {
                            computeQueue.push_back({workload, callback_t([this, numWrite, writeAddr]() {
                                for (size_t j = 0; j < numWrite; j++) {
                                    writeQueue.push_back({ pointer_t(writeAddr + j * config.BLOCK_SIZE), callback_t([]() {}) });
                                }
                            })});
                        }
                    }) });
                }
            }
        }

        PE::clockUpdate();
    }

    bool done() {
        return !dispatcher.haveTask(peid) && readQueue.empty() && computeQueue.empty() && writeQueue.empty();
    }

    virtual void printStats() {
        // return;
        fprintf(stderr, "PE %zu: Dispatched %zu tasks\n", peid, dispatchedTasks);
        fprintf(stderr, "PE %zu: Queue R %zu C %zu W %zu\n", peid, readQueue.size(), computeQueue.size(), writeQueue.size());
    }

private:
    TaskDispatcher<MergeTask> &dispatcher;
    size_t peid;

    size_t dispatchedTasks = 0;
};

class DRAMBackend : public Module {
private:
    using Tech = ramulator::HBM;
    using Memory = ramulator::Memory<Tech, ramulator::Controller>;

    static constexpr size_t NUM_OUTSTANDING_REQUESTS = 128;
    static constexpr double DRAM_CLOCK_PERIOD = 1.5;
    static constexpr size_t DRAM_GRANULARITY = 32;

public:
    DRAMBackend(size_t channelWidth, std::string name) : channelWidth(channelWidth), memFIFO(MemoryPortFIFO::DEFAULT_CAPACITY, ~0), memport(memFIFO.getSlavePort()), name(name) {
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
        if (cntPendingResponse > 0) {
            cntPendingResponse--;
            memport.resp.write({ MemoryResponse{} });
        }

        if (memport.req.isReadable() && cntPendingReqs < NUM_OUTSTANDING_REQUESTS) {
            MemoryRequest req = memport.req.read(1)[0];
            long translatedAddr = addrTranslator(req.addr);

            if (req.is_write)
                cntWrite++;
            else
                cntRead++;

            for (size_t offset = 0; offset < channelWidth; offset += DRAM_GRANULARITY) {
                bool isLast = offset + DRAM_GRANULARITY >= channelWidth;
                ramulator::Request ramReq(translatedAddr + offset, req.is_write ? ramulator::Request::Type::WRITE : ramulator::Request::Type::READ, [this, isLast](ramulator::Request &r) {
                    if (isLast)
                        cntPendingResponse++;
                    cntPendingReqs--;
                });
                cntPendingReqs++;
                requests.push(std::move(ramReq));
            }
        }

        if (cntClock >= DRAM_CLOCK_PERIOD) {
            cntClock -= DRAM_CLOCK_PERIOD;

            if (!requests.empty()) {
                if (memory->send(requests.front()))
                    requests.pop();
            }

            memory->tick();
        }
        cntClock += 1;
    }
    virtual void clockApply() override {}

    virtual void printStats() {
        fprintf(stderr, "DRAM %s: READ %zu WRITE %zu, PENDING REQ %zu, BANDWIDTH %.3lf B/cycle\n", name.c_str(), cntRead, cntWrite, cntPendingReqs, double(cntRead + cntWrite) * channelWidth / Module::cntCycle);
    }


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

    size_t cntPendingResponse = 0;

    std::string name;
    size_t cntRead = 0, cntWrite = 0;
};

size_t simulateOuterSPACEMultiply(TaskProvider &provider) {
    TaskDispatcherStatic<MultiplyTask> dispatcherMultiply(provider.getMultiplyTasks(), config.NUM_PE);

    assert(config.NUM_PE % config.NUM_PE_TILES == 0);
    const size_t NUM_PE_PER_TILE = config.NUM_PE / config.NUM_PE_TILES;

    std::vector<std::vector<std::unique_ptr<PEMultiplier>>> pes(config.NUM_PE_TILES);
    std::vector<std::unique_ptr<Crossbar>> crossbar(config.NUM_PE_TILES);
    std::vector<std::vector<std::unique_ptr<Cache>>> cache(config.NUM_PE_TILES);
    std::unique_ptr<Crossbar> gcrossbar;
    std::vector<std::unique_ptr<DRAMBackend>> dram(config.NUM_DRAM_CHANNELS);

    gcrossbar.reset(new Crossbar(config.NUM_PE, config.NUM_DRAM_CHANNELS, "Global"));

    for (size_t i = 0; i < config.NUM_PE_TILES; i++) {
        crossbar[i].reset(new Crossbar(NUM_PE_PER_TILE, NUM_PE_PER_TILE, std::string("Tile ") + std::to_string(i)));
        
        pes[i].resize(NUM_PE_PER_TILE);
        for (size_t j = 0; j < NUM_PE_PER_TILE; j++) {
            pes[i][j].reset(new PEMultiplier(dispatcherMultiply, i * NUM_PE_PER_TILE + j));
            pes[i][j]->setMemoryPort(crossbar[i]->getUpstreamPort(j));
        }

        cache[i].resize(NUM_PE_PER_TILE);
        for (size_t j = 0; j < NUM_PE_PER_TILE; j++) {
            cache[i][j].reset(new Cache(SimCache(config.L0_CACHE_NUM_GROUPS, clog2(config.L0_CACHE_SIZE / config.BLOCK_SIZE / config.L0_CACHE_NUM_GROUPS / NUM_PE_PER_TILE), clog2(config.BLOCK_SIZE)), std::to_string(i) + "-" + std::to_string(j)));
            crossbar[i]->setDownstreamPort(j, cache[i][j]->getUpstreamPort());
            cache[i][j]->setDownstreamPort(gcrossbar->getUpstreamPort(i * NUM_PE_PER_TILE + j));
        }

        crossbar[i]->setMapper([NUM_PE_PER_TILE](pointer_t addr) -> size_t {
            return addr / (config.L0_CACHE_SIZE / NUM_PE_PER_TILE) % NUM_PE_PER_TILE;
        });
    }

    for (size_t i = 0; i < config.NUM_DRAM_CHANNELS; i++) {
        dram[i].reset(new DRAMBackend(config.BLOCK_SIZE, std::to_string(i)));
        dram[i]->setAddressTranslator([](pointer_t addr) -> long {
            return addr / config.BLOCK_SIZE / config.NUM_DRAM_CHANNELS * config.BLOCK_SIZE + addr % config.BLOCK_SIZE;
        });
        gcrossbar->setDownstreamPort(i, dram[i]->getPort());
    }

    constexpr size_t PAGE_SIZE = 4096;

    gcrossbar->setMapper([](pointer_t addr) -> size_t {
        return addr / PAGE_SIZE % config.NUM_DRAM_CHANNELS;
    });

    size_t cnt = 0;
    while (true) {
        Module::updateAll();
        cnt++;

        if (cnt % 100000 == 0)
        {
            fprintf(stderr, "=== STATS AT CYCLE %8zu ===\n", cnt);
            Module::printStatsAll();
            fprintf(stderr, "===============================\n");
        }

        bool alldone = true;
        for (auto &&tile : pes) {
            for (auto &&pe : tile) {
                if (!pe->done()) {
                    alldone = false;
                    break;
                }
            }
        }

        if (alldone) {
            break;
        }
    }

    return cnt;
}

size_t simulateOuterSPACEMerge(TaskProvider &provider) {
    TaskDispatcherStatic<MergeTask> dispatcherMerge(provider.getMergeTasks(), config.NUM_PE);

    assert(config.NUM_PE % config.NUM_PE_TILES == 0);
    const size_t NUM_PE_PER_TILE = config.NUM_PE / config.NUM_PE_TILES;

    std::vector<std::unique_ptr<PEMerger>> pes(config.NUM_PE);
    std::unique_ptr<Crossbar> gcrossbar;
    std::vector<std::unique_ptr<DRAMBackend>> dram(config.NUM_DRAM_CHANNELS);

    gcrossbar.reset(new Crossbar(config.NUM_PE, config.NUM_DRAM_CHANNELS, "Global"));

    for (size_t i = 0; i < config.NUM_PE; i++) {
        pes[i].reset(new PEMerger(dispatcherMerge, i));
        pes[i]->setMemoryPort(gcrossbar->getUpstreamPort(i));
    }

    for (size_t i = 0; i < config.NUM_DRAM_CHANNELS; i++) {
        dram[i].reset(new DRAMBackend(config.BLOCK_SIZE, std::to_string(i)));
        dram[i]->setAddressTranslator([](pointer_t addr) -> long {
            return addr / config.BLOCK_SIZE / config.NUM_DRAM_CHANNELS * config.BLOCK_SIZE + addr % config.BLOCK_SIZE;
            });
        gcrossbar->setDownstreamPort(i, dram[i]->getPort());
    }

    constexpr size_t PAGE_SIZE = 4096;

    gcrossbar->setMapper([](pointer_t addr) -> size_t {
        return addr / PAGE_SIZE % config.NUM_DRAM_CHANNELS;
    });

    size_t cnt = 0;
    while (true) {
        Module::updateAll();
        cnt++;

        if (cnt % 100000 == 0)
        {
            fprintf(stderr, "=== STATS AT CYCLE %8zu ===\n", cnt);
            Module::printStatsAll();
            fprintf(stderr, "===============================\n");
        }

        bool alldone = true;
        for (auto &&pe : pes) {
            if (!pe->done()) {
                alldone = false;
                break;
            }
        }

        if (alldone) {
            break;
        }
    }

    return cnt;
}

size_t simulateOuterSPACE(const CSRMatrix &lmatCSC, const CSRMatrix &rmatCSR) {
    TaskProvider provider(lmatCSC, rmatCSR);

    size_t cycleMultiplyAnalytical = simulateOuterSPACEAnalyticalMultiply(provider);
    fprintf(stderr, "OuterSPACE Multiply Analytical: %zu\n", cycleMultiplyAnalytical);

    size_t cycleMultiplyCycleAccurate = simulateOuterSPACEMultiply(provider);
    fprintf(stderr, "OuterSPACE Multiply Cycle-Accurate: %zu\n", cycleMultiplyCycleAccurate);

    size_t cycleMergeAnalytical = simulateOuterSPACEAnalyticalMerge(provider);
    fprintf(stderr, "OuterSPACE Merge Analytical: %zu\n", cycleMergeAnalytical);

    size_t cycleMergeCycleAccurate = simulateOuterSPACEMerge(provider);
    fprintf(stderr, "OuterSPACE Merge Cycle-Accurate: %zu\n", cycleMergeCycleAccurate);

    return cycleMultiplyCycleAccurate + cycleMergeCycleAccurate;
}