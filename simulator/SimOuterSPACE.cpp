#include "common.h"
#include <cassert>
#include <iterator>
#include <algorithm>

#include <deque>

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
class TaskDispatcherStatic {
public:
    TaskDispatcherStatic(const std::vector<T> &tasks, size_t numPEs) : peTasks(numPEs) {
        size_t taskPerPE = (tasks.size() + numPEs - 1) / numPEs;

        for (size_t i = 0; i < tasks.size(); i++) {
            peTasks[i / taskPerPE].push_back(tasks[i]);
        }
    }

    bool haveTask(size_t peid) {
        return !peTasks[peid].empty();
    }
    T nextTask(size_t peid) {
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