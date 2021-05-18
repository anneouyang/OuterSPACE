#include <cstdio>
#include <cstdint>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <deque>
#include <queue>
#include <set>
#include <map>

// #include <execution>

#include "common.h"
// #include "SimCache.h"

#undef NDEBUG
#include <cassert>

class Timer
{
public:
	Timer(std::string caption) : caption(std::move(caption)), timeStart(std::chrono::high_resolution_clock::now()) {}
	~Timer()
	{
		auto timeStop = std::chrono::high_resolution_clock::now();
		std::cout << " -- " << caption << ": " << std::chrono::duration_cast<std::chrono::duration<double>>(timeStop - timeStart).count() << " s" << std::endl;
	}

protected:
	std::chrono::time_point<std::chrono::high_resolution_clock> timeStart;
	std::string caption;
};

#define TIMER(caption) if(Timer timer(caption); true)




void dupcheck(const COOMatrix &input)
{
	for (size_t i = 0; i + 1 < input.size(); i++)
	{
		if (input[i].row == input[i + 1].row && input[i].col == input[i + 1].col)
		{
			throw(233);
			assert(false);
		}
	}
}

COOMatrix readcoo(std::istream &in, size_t &NRow, size_t &NCol, bool sym)
{
	NRow = NCol = 0;
	size_t NNZ = 0;

	COOMatrix result;

	bool firstline = true;

	for(std::string line; std::getline(in, line); )
	{
		bool skip = true;
		for (size_t i = 0; i < line.size(); i++)
		{
			if (line[i] == ' ' || line[i] == '\t')
				continue;
			if (line[i] == '%')
				break;
			skip = false;
			break;
		}
		if (skip)
			continue;

		if (firstline)
		{
			sscanf(line.c_str(), "%zu %zu %zu", &NRow, &NCol, &NNZ);
			if (sym)
				result.reserve(NNZ * 2);
			else
				result.reserve(NNZ);
			firstline = false;
			continue;
		}

		size_t row, col;
		double val;
		if (sscanf(line.c_str(), "%zu %zu %lf", &row, &col, &val) < 3)
			val = 1.0;
		result.push_back(COOElement{ index_t(row - 1), index_t(col - 1), value_t(val) });
		if(sym && row != col)
			result.push_back(COOElement{ index_t(col - 1), index_t(row - 1), value_t(val) });
	}
	
	return result;
}

template<bool transpose = false>
CSRMatrix coo2csr(COOMatrix coo, size_t N)
{
	CSRMatrix result;
	result.pos.resize(N + 1, 0);
	result.data.resize(coo.size());
	/*result.idx.resize(coo.size());
	result.val.resize(coo.size());*/

	if constexpr (transpose)
	{
		std::sort(/* std::execution::par_unseq,  */ coo.begin(), coo.end(), [](const COOElement &a, const COOElement &b)
		{
			return a.col == b.col ? a.row < b.row : a.col < b.col;
		});
	}
	else
	{
		std::sort(/* std::execution::par_unseq,  */ coo.begin(), coo.end());
	}

	dupcheck(coo);

	index_t lastIdx = ~0;
	size_t  lastPos = 0;

	for (COOElement e : coo)
	{
		if constexpr (transpose)
			std::swap(e.row, e.col);
		while (e.row != lastIdx)
		{
			lastIdx++;
			result.pos[lastIdx] = lastPos;
		}
		result.data[lastPos] = CSRElement{ e.col, e.val };
		/*result.idx[lastPos] = e.col;
		result.val[lastPos] = e.val;*/
		lastPos++;
	}

	for (auto iter = result.pos.rbegin(); iter != result.pos.rend(); iter++)
	{
		if (*iter != 0)
			break;
		*iter = result.data.size();
	}
	//result.pos.back() = result.idx.size();

	return result;
}

CompactCOOMatrix csr2compact(const CSRMatrix &csr)
{
	if (csr.pos.empty())
		return CompactCOOMatrix();

	CompactCOOMatrix result;
	result.data.resize(csr.data.size());

	TIMER("CSR2Compact")
	{

		size_t maxNNZR = 0;
		for (size_t i = 0; i + 1 < csr.pos.size(); i++)
		{
			maxNNZR = std::max(maxNNZR, csr.pos[i + 1] - csr.pos[i]);
		}

		// statNNZR[i] = # rows with nnzr == i + 1
		std::vector<size_t> statNNZR(maxNNZR);
		for (size_t i = 0; i + 1 < csr.pos.size(); i++)
		{
			size_t nnzr = csr.pos[i + 1] - csr.pos[i];
			if (nnzr != 0)
				statNNZR[nnzr - 1]++;
		}

		// statNNZR[i] = # rows with nnzr >= i + 1
		for (size_t i = statNNZR.size() - 1; i > 0; i--)
		{
			statNNZR[i - 1] += statNNZR[i];
		}


		result.pos.resize(maxNNZR + 1);
		/*for (size_t i = 0; i < maxNNZR; i++)
			result[i].resize(statNNZR[i]);*/

		for (size_t i = 1; i <= maxNNZR; i++)
			result.pos[i] = result.pos[i - 1] + statNNZR[i - 1];

		std::vector<size_t> lenResult(maxNNZR, 0);


		for(size_t i = 0; i + 1 < csr.pos.size(); i++)
		{
			size_t nnzr = csr.pos[i + 1] - csr.pos[i];


			const CSRElement *element_start = &csr.data[csr.pos[i]];
			/*const index_t *idx_start = &csr.idx[csr.pos[i]];
			const value_t *val_start = &csr.val[csr.pos[i]];*/

			for (size_t j = 0; j < nnzr; j++)
			{
				// result.data[result.pos[j] + lenResult[j]] = COOElement{ index_t(i), element_start[j].idx, element_start[j].val };
				result.data[result.pos[j] + lenResult[j]] = COOElement{ index_t(i), element_start[j] };

				lenResult[j]++;
			}

		}

	}

	return result;
}

CompactCOOMatrix csc2rawcompact(const CSRMatrix &csc)
{
	CompactCOOMatrix result;
	result.pos.resize(csc.pos.size());
	result.data.resize(csc.data.size());

	for (size_t i = 0; i < csc.pos.size(); i++)
		result.pos[i] = csc.pos[i];

	index_t colid = 0;

	for (size_t i = 0; i < csc.data.size(); i++)
	{
		while (i >= csc.pos[colid + 1])
			colid++;
		result.data[i].row = csc.data[i].idx;
		result.data[i].col = colid;
		result.data[i].val = csc.data[i].val;
	}

	return result;
}




std::vector<COOMatrix> compactMulcsr(const CompactCOOMatrix &compact, const CSRMatrix &csr)
{
	std::vector<COOMatrix> result(compact.pos.size() - 1);
	for (size_t i = 0; i + 1 < compact.pos.size(); i++)
	{
		for (size_t j = compact.pos[i]; j < compact.pos[i + 1]; j++)
		{
			size_t row = compact.data[j].col;
			for (size_t k = csr.pos[row]; k < csr.pos[row + 1]; k++)
			{
				result[i].push_back(COOElement{ compact.data[j].row, csr.data[k].idx, csr.data[k].val * compact.data[j].val });
			}
		}
		dupcheck(result[i]);
	}
	return result;
}

std::vector<COOMatrix> cscMulcsr(const CSRMatrix &csc, const CSRMatrix &csr)
{
	assert(csc.pos.size() == csr.pos.size());
	std::vector<COOMatrix> result;
	for (size_t i = 0; i + 1 < csc.pos.size(); i++)
	{
		if (csc.pos[i] == csc.pos[i + 1] || csr.pos[i] == csr.pos[i + 1])
			continue;
		result.emplace_back();
		for(size_t j = csc.pos[i]; j < csc.pos[i + 1]; j++)
			for (size_t k = csr.pos[i]; k < csr.pos[i + 1]; k++)
			{
				result.back().push_back(COOElement{ csc.data[j].idx, csr.data[k].idx, csc.data[j].val * csr.data[k].val });
			}
	}
	return result;
}

bool compareCOO(COOMatrix a, COOMatrix b, value_t eps = 1e-6)
{
    if (a.size() != b.size())
        return false;
    std::sort(/* std::execution::par_unseq,  */ a.begin(), a.end());
    std::sort(/* std::execution::par_unseq,  */ b.begin(), b.end());
    for (size_t i = 0; i < a.size(); i++)
    {
        if (a[i].row != b[i].row || a[i].col != b[i].col)
            return false;
        if (abs(a[i].val - b[i].val) > eps)
            return false;
    }
    return true;
}

bool sanityCompactCOO(const CompactCOOMatrix &compact, const COOMatrix &original)
{
    return compareCOO(compact.data, original);
}

#if 0

COOMatrix merge2way(const COOMatrix &a, const COOMatrix &b)
{
	/*dupcheck(a);
	dupcheck(b);*/

	COOMatrix result;
	size_t i = 0, j = 0;
	while (i < a.size() && j < b.size())
	{
		if (a[i] < b[j])
			result.push_back(a[i]), i++;
		else if (b[j] < a[i])
			result.push_back(b[j]), j++;
		else
			result.push_back(COOElement(a[i].row, a[i].col, a[i].val + b[j].val)), i++, j++;
	}
	while (i < a.size())
		result.push_back(a[i]), i++;
	while (j < b.size())
		result.push_back(b[j]), j++;
	return result;
}

constexpr size_t MERGE_LAYER = 6;
constexpr size_t MAX_MERGE_K = 1 << MERGE_LAYER;

size_t workload[MERGE_LAYER];

SimCache cache(8, 13, 3);

struct MixedCOOMatrix
{
	COOMatrix mat;
	bool isMulResult;
};

std::vector<size_t> simRowOrder;

void accessRow(const CSRMatrix &csr, size_t rowid)
{
	cache.access(uintptr_t(&csr.pos[rowid]));
	cache.access(uintptr_t(&csr.pos[rowid + 1]));
	for (size_t i = csr.pos[rowid]; i < csr.pos[rowid + 1]; i++)
	{
		cache.access(uintptr_t(&csr.data[i].idx));
		cache.access(uintptr_t(&csr.data[i].val));
		cache.access(uintptr_t(&csr.data[i].val) + 4);
	}
	// printf("ACCESS ROW %d\n", rowid);
	simRowOrder.push_back(rowid);
}

std::vector<COOMatrix> multHardware(std::vector<MixedCOOMatrix> rawInput, const CSRMatrix &csr)
{
	std::vector<COOMatrix> result(rawInput.size());

	std::vector<std::deque<COOElement>> inputNonMulted(rawInput.size());

	auto priorityLess = [&](size_t lhs, size_t rhs) -> bool
	{
		assert(!rawInput[lhs].isMulResult && !inputNonMulted[lhs].empty());
		assert(!rawInput[rhs].isMulResult && !inputNonMulted[rhs].empty());
		return inputNonMulted[lhs].front().row > inputNonMulted[rhs].front().row;
	};
	std::priority_queue<size_t, std::vector<size_t>, decltype(priorityLess)> pq(priorityLess);

	for (size_t i = 0; i < rawInput.size(); i++)
	{
		if (rawInput[i].isMulResult)
			result[i] = std::move(rawInput[i].mat);
		else
		{
			std::copy(rawInput[i].mat.begin(), rawInput[i].mat.end(), std::back_inserter(inputNonMulted[i]));
			if(!rawInput[i].mat.empty())
				pq.push(i);
		}
	}

	constexpr size_t BLOCK_SIZE = 4;

	while (!pq.empty())
	{
		size_t idx = pq.top();
		pq.pop();

		/*for (size_t k = 0; k < BLOCK_SIZE && !inputNonMulted[idx].empty(); k++)
		{*/
			COOElement e = inputNonMulted[idx].front();
			inputNonMulted[idx].pop_front();

			accessRow(csr, e.col);

			for (size_t i = csr.pos[e.col]; i < csr.pos[e.col + 1]; i++)
			{
				result[idx].push_back(COOElement{ e.row, csr.data[i].idx, e.val * csr.data[i].val });
			}
		//}

		if (!inputNonMulted[idx].empty())
			pq.push(idx);
	}

	return result;
}

COOMatrix mergeHardware(std::vector<MixedCOOMatrix> rawInput, const CSRMatrix &csr)
{
	std::vector<COOMatrix> input, output;

	input = multHardware(std::move(rawInput), csr);

	for (auto &&mat : input)
	{
		dupcheck(mat);
	}

	for (size_t layer = 0; layer < MERGE_LAYER; layer++)
	{
		for (size_t i = 0; i + 1 < input.size(); i += 2)
		{
			workload[layer] += input[i].size() + input[i + 1].size();
			output.push_back(merge2way(input[i], input[i + 1]));
		}
		if (input.size() % 2 == 1)
		{
			workload[layer] += input.back().size();
			output.push_back(std::move(input.back()));
		}

		input = std::move(output);
	}

	assert(input.size() == 1);
	
	return input[0];
}

size_t cntMemAccess = 0;

COOMatrix merge(const CompactCOOMatrix &compact, const CSRMatrix &csr)
{
	if (compact.pos.size() <= 1)
		return COOMatrix();

	size_t mergeWay = compact.pos.size() - 1;
	
	float nnzr = float(csr.data.size()) / csr.NRow();
	/*if (mergeWay == 1)
		return input[0];*/

	size_t mergeK = mergeWay > MAX_MERGE_K ? mergeWay % (MAX_MERGE_K - 1) : mergeWay;

	std::deque<MixedCOOMatrix> q1, q2;

	for (size_t i = 0; i + 1 < compact.pos.size(); i++)
	{
		MixedCOOMatrix tmp;
		std::copy(compact.data.begin() + compact.pos[i], compact.data.begin() + compact.pos[i + 1], std::back_inserter(tmp.mat));
		tmp.isMulResult = false;
		q1.push_back(std::move(tmp));
	}

	// std::move(input.begin(), input.end(), std::back_inserter(q1));
	std::sort(/* std::execution::par_unseq,  */ q1.begin(), q1.end(), [](const MixedCOOMatrix &lhs, const MixedCOOMatrix &rhs) { return lhs.mat.size() < rhs.mat.size(); });

	std::vector<MixedCOOMatrix> tmp;
	while (!q1.empty() || !q2.empty())
	{
		bool nextQ1;
		if (q1.empty())
			nextQ1 = false;
		else if (q2.empty())
			nextQ1 = true;
		else if (size_t(q1.front().mat.size() * nnzr) < q2.front().mat.size())
			nextQ1 = true;
		else
			nextQ1 = false;

		if (nextQ1)
		{
			tmp.push_back(std::move(q1.front()));
			q1.pop_front();
		}
		else
		{
			cntMemAccess += q2.front().mat.size();
			tmp.push_back(std::move(q2.front()));
			q2.pop_front();
		}

		assert(tmp.size() <= mergeK);

		if (tmp.size() == mergeK)
		{
			
			COOMatrix result = mergeHardware(std::move(tmp), csr);

			if (q1.empty() && q2.empty())
				return result;

			cntMemAccess += result.size();
			q2.push_back(MixedCOOMatrix{ std::move(result), true });

			tmp.clear();

			mergeK = MAX_MERGE_K;
		}
	}

	assert(false);
	return COOMatrix();
}

COOMatrix deduplicateCOO(COOMatrix coo)
{
	std::sort(/* std::execution::par_unseq,  */ coo.begin(), coo.end());

	COOMatrix result;
	result.push_back(coo.front());

	for (size_t i = 1; i < coo.size(); i++)
	{
		if (coo[i].row != coo[i - 1].row || coo[i].col != coo[i - 1].col)
			result.push_back(coo[i]);
		else
			result.back().val += coo[i].val;
	}

	return result;
}




std::vector<size_t> generateAccessOrder(const CSRMatrix &csr)
{
	std::vector<size_t> rowAccessOrder;

	for (size_t i = 0; i + 1 < csr.pos.size(); i++)
	{
		for (size_t j = csr.pos[i]; j < csr.pos[i + 1]; j++)
			rowAccessOrder.push_back(csr.data[j].idx);
	}

	return rowAccessOrder;
}

std::vector<std::list<size_t>> computeNextUse(const std::vector<size_t> &order, size_t NRow)
{
	std::vector<std::list<size_t>> nextUse(NRow);
	for (size_t i = 0; i < order.size(); i++)
		nextUse[order[i]].push_back(i);
	return nextUse;
}

size_t policyMIN(const CSRMatrix &csr, const std::vector<size_t> &order, size_t cacheSize, size_t lineSize)
{
	auto nextUse = computeNextUse(order, csr.NRow());
	std::vector<bool> inCache(csr.NRow(), false);

	std::set<size_t> cacheNextUse;	// contains nextuse of all rows in cache
	size_t cacheUtil = 0;
	size_t memAccess = 0;

	auto getSize = [&](size_t rowId) -> size_t
	{
		return (sizeof(index_t) + sizeof(value_t)) * (csr.pos[rowId + 1] - csr.pos[rowId]) 
			+ sizeof(index_t);	// +4 for size
	};

	auto pushCache = [&](size_t rowId)
	{
		assert(!inCache[rowId]);

		if (nextUse[rowId].empty())
			return;
		inCache[rowId] = true;
		cacheNextUse.insert(nextUse[rowId].front());

		if (lineSize != 0)
			cacheUtil += lineSize;
		else
			cacheUtil += getSize(rowId);
	};

	auto popCache = [&](size_t rowId)
	{
		if (!inCache[rowId])
			return;

		inCache[rowId] = false;
		cacheNextUse.erase(nextUse[rowId].front());

		if (lineSize != 0)
			cacheUtil -= lineSize;
		else
			cacheUtil -= getSize(rowId);
	};

	for (size_t i = 0; i < order.size(); i++)
	{
		size_t rowId = order[i];
		size_t sizeRow = getSize(rowId);

		if (inCache[rowId])
		{
			popCache(rowId);
			if (lineSize != 0 && sizeRow > lineSize)
			{
				memAccess += sizeof(index_t);
				memAccess += sizeRow - lineSize;
			}
		}
		else
		{
			memAccess += sizeof(index_t);	// for position in csr.data
			memAccess += sizeRow;
		}

		nextUse[rowId].pop_front();

		if (sizeRow > cacheSize || nextUse[rowId].empty())
			continue;

		pushCache(rowId);

		std::vector<size_t> victims; // excluding current row

		while (cacheUtil > cacheSize)
		{
			size_t victimRow = order[*cacheNextUse.rbegin()];
			popCache(victimRow);

			if(victimRow != rowId)
				victims.push_back(victimRow);
		}

		if (!inCache[rowId])	// Oops, we popped the current row. Recover victims
		{
			for (size_t victimRow : victims)
				pushCache(victimRow);
		}

		assert(cacheUtil <= cacheSize);
	}

	return memAccess;
}



size_t policySlotMIN(const CSRMatrix &csr, const std::vector<size_t> &order, size_t numSlot, size_t sizeSlot, size_t predictWindow)
{
	struct memOpTag
	{
		size_t rowid;
		size_t partid;

		bool operator<(const memOpTag &rhs) const
		{
			return rowid == rhs.rowid ? partid < rhs.partid : rowid < rhs.rowid;
		}
		bool operator==(const memOpTag &rhs) const
		{
			return rowid == rhs.rowid && partid == rhs.partid;
		}
	};
	struct memOp
	{
		memOpTag tag;
		size_t memLength;
		size_t nextUse;
	};

	std::list<memOp> memOpWindow;

	size_t orderId = 0, finishLen = 0;

	std::map<memOpTag, memOp *> lastUse;

	auto generateMemOp = [&]() -> bool
	{
		while (true)
		{
			if (orderId >= order.size())
				return false;

			size_t rowid = order[orderId];
			size_t len = (csr.pos[rowid + 1] - csr.pos[rowid]) * (sizeof(index_t) + sizeof(value_t)) + sizeof(index_t);

			if (finishLen >= len)
			{
				orderId++;
				finishLen = 0;
			}
			else
			{
				memOp op;
				op.tag = { rowid, finishLen / sizeSlot };
				op.memLength = len - finishLen > sizeSlot ? sizeSlot : len - finishLen;

				if (finishLen == 0)
					op.memLength += sizeof(index_t);

				op.nextUse = ~0 - 1;
				memOpWindow.push_back(op);
				finishLen += sizeSlot;

				if (lastUse.count(op.tag))
				{
					lastUse[op.tag]->nextUse = orderId;
				}
				lastUse[op.tag] = &memOpWindow.back();
				return true;
			}
		}
		
	};

	auto nextMemOp = [&]() -> std::tuple<bool, memOp>
	{
		while (memOpWindow.size() < predictWindow)
		{
			if (!generateMemOp())
				break;
		}

		if (memOpWindow.empty())
			return { false, memOp() };

		auto result = memOpWindow.front();
		
		assert(lastUse.count(result.tag));
		if (lastUse[result.tag] == &memOpWindow.front())
			lastUse.erase(result.tag);

		memOpWindow.pop_front();
		return { true, result };
	};

	const size_t roundnumSlot = size_t(1) << clog2(numSlot);

	struct maxWeight
	{
		size_t idx;
		size_t weight;

		bool operator<(const maxWeight &rhs) const { return weight < rhs.weight; }
	};

	std::vector<maxWeight> maxWeights(2 * roundnumSlot);
	for (size_t i = 0; i < roundnumSlot; i++)
		maxWeights[roundnumSlot + i] = maxWeight{ i, size_t(~0) };
	for (size_t i = roundnumSlot - 1; i > 0; i--)
		maxWeights[i] = std::max(maxWeights[i * 2], maxWeights[i * 2 + 1]);

	auto updateWeight = [&](size_t idx, size_t val)
	{
		maxWeights[roundnumSlot + idx] = maxWeight{ idx, val };
		for (size_t i = (roundnumSlot + idx) / 2; i > 0; i /= 2)
			maxWeights[i] = std::max(maxWeights[i * 2], maxWeights[i * 2 + 1]);
	};

	const memOpTag invalidTag = memOpTag{ size_t(~0), size_t(~0) };
	std::vector<memOpTag> cacheTag(numSlot, invalidTag);
	std::map<memOpTag, size_t> inCache;		// tag -> position in cache

	size_t memAccess = 0, cacheRef = 0;

	while (true)
	{
		auto [succ, op] = nextMemOp();
		if (!succ)
			break;

		cacheRef += op.memLength;

		if (inCache.count(op.tag))
		{
			updateWeight(inCache[op.tag], op.nextUse);
			continue;
		}

		memAccess += op.memLength;

		

		size_t victimIdx = maxWeights[1].idx;
		memOpTag victim = cacheTag[victimIdx];
		if (maxWeights[1].weight < op.nextUse)
			continue;

		if (!(victim == invalidTag))
		{
			inCache.erase(victim);
		}
		cacheTag[victimIdx] = op.tag;
		updateWeight(victimIdx, op.nextUse);
		inCache[op.tag] = victimIdx;
	}

	// printf("SLOT-MIN CACHE MISS = %zu / %zu (%.3lf)\n", memAccess, cacheRef, double(memAccess) / cacheRef);

	return memAccess;
}

#endif

size_t simulateCycle(const CompactCOOMatrix &lmat, const CSRMatrix &rmat, std::string dumpFilePrefix);
size_t simulateOuterSPACEAnalytical(const CSRMatrix &lmatCSC, const CSRMatrix &rmatCSR);
size_t simulateOuterSPACE(const CSRMatrix &lmatCSC, const CSRMatrix &rmatCSR);
// COOMatrix reorderGraph(COOMatrix input);

int main(int argc, char *argv[])
{
	bool SYMMETRIC = false;
	std::string filename[2];

	filename[0] = argv[1];
	filename[1] = argv[2];

	/*if (argc > 1)
	{
		filename = argv[1];
		if (argc > 2 && std::string(argv[2]) == "1")
			SYMMETRIC = true;
		else
			SYMMETRIC = false;
	}
	else
	{
		filename = R"(D:\Project\SparseAccelerator\simulator\SimSpGEMM\gcc\rmat\rmat-10k-x8.mtx)";
		SYMMETRIC = false;
	}*/

	size_t NRow[2], NCol[2];
	COOMatrix coo[2];
	
	TIMER("Read Matrix")
	{
		for (size_t i = 0; i < 2; i++) {
			std::ifstream fin(filename[i]);
			coo[i] = readcoo(fin, NRow[i], NCol[i], SYMMETRIC);
		}
	}

	// Workaround: Transpose Matrix 2
	std::swap(NRow[1], NCol[1]);
	for (auto &&e : coo[1]) {
		std::swap(e.row, e.col);
	}
	//

	/*TIMER("Gorder")
	{
		coo = reorderGraph(std::move(coo));
	}*/

	for (size_t i = 0; i < 2; i++) {
		size_t nnz = coo[i].size();
		printf("NCol = %zu, NRow = %zu, NNZ = %zu\n", NRow[i], NCol[i], nnz);
	}

	/*if (NRow != NCol)
	{
		printf("WARNING: NRow != NCol\n");
	}*/

	CSRMatrix csr, csc;
	
	TIMER("COO2CSR")
	{
		csc = coo2csr<true>(coo[0], NCol[0]);
		csr = coo2csr(coo[1], NRow[1]);
	}

	assert(csr.pos.size() == csc.pos.size());

	size_t mulflops_ref = 0;
	for (size_t i = 0; i + 1 < csr.pos.size(); i++)
	{
		size_t nnzc = csc.pos[i + 1] - csc.pos[i];
		size_t nnzr = csr.pos[i + 1] - csr.pos[i];
		mulflops_ref += nnzc * nnzr;
	}
	printf("mul flops ref = %zu\n", mulflops_ref);

	size_t cycleOuterSPACE = simulateOuterSPACE(csc, csr);
	printf("OuterSPACE cycles = %zu Perf=%.2lf GFlops\n", cycleOuterSPACE, (double)mulflops_ref / cycleOuterSPACE * 1.5);
	
#if 0

	CompactCOOMatrix compact;
	
	compact = csr2compact(csr);
	// compact = csc2rawcompact(csc);

	if (!sanityCompactCOO(compact, coo))
	{
		printf("Failed sanity check! Compact COO != Original COO\n");
		return 1;
	}

	printf("# compact col = %zu\n", compact.pos.size() - 1);
	printf("# nnz = %zu\n", compact.data.size());
	printf("nnzr = ");
	for (size_t i = 0; i + 1 < compact.pos.size(); i++)
	{
		printf("%zu ", compact.pos[i+1] - compact.pos[i]);
	}
	printf("\n");

#if 0

	size_t mulflops_sum = 0;
	printf("mult. flops = ");
	for (size_t i = 0; i + 1 < compact.pos.size(); i++)
	{
		size_t mulflops = 0;
		for (size_t j = compact.pos[i]; j < compact.pos[i+1]; j++)
		{
			COOElement &e = compact.data[j];
			size_t nnzr = csr.pos[e.col + 1] - csr.pos[e.col];
			mulflops += nnzr;
		}
		mulflops_sum += mulflops;
		printf("%zu ", mulflops);
	}
	printf("\n");
	printf("mult. flops sum = %zu\n", mulflops_sum);

	auto mulresult = compactMulcsr(compact, csr);
	// auto mulresult = cscMulcsr(csc, csr);

	size_t sizeRaw = 0;
	for (auto &mat : mulresult)
		sizeRaw += mat.size();
	printf("mult. result size = %zu\n", sizeRaw);

#endif

#if 0
	auto finalresult = merge(compact, csr);

	printf("after deduplicate size = %zu\n", finalresult.size());

	printf("memory access = %zu\n", cntMemAccess);

	printf("workload = %zu", workload[0]);
	for (size_t i = 1; i < MERGE_LAYER; i++)
		printf(",%zu", workload[i]);
	printf("\n");

	printf("Cache Ref Bytes = %zu\n", cache.getNumRef() * 4);
	printf("Cache Miss Bytes = %zu\n", cache.getNumMiss() * 8);
#endif

#if 0

	printf("MEM read using row-by-row order = %zu\n", policyMIN(csr, generateAccessOrder(csr), 512 * 1024, 0));
	printf("MEM read using MIN policy = %zu\n", policyMIN(csr, simRowOrder, 512 * 1024, 0));
	printf("MEM read using fixed line size = %zu\n", policyMIN(csr, simRowOrder, 512 * 1024, 512 + 4));

	printf("MEM read using slot-based MIN = %zu\n", policySlotMIN(csr, simRowOrder, 4096, 256, ~0));

	/*printf("MEM read w/ Cache Size Ablation (256*1024) = %zu\n", policySlotMIN(csr, simRowOrder, 256, 1024, 8192));
	printf("MEM read w/ Cache Size Ablation (512*512) = %zu\n", policySlotMIN(csr, simRowOrder, 512, 512, 8192));
	printf("MEM read w/ Cache Size Ablation (1024*256) = %zu\n", policySlotMIN(csr, simRowOrder, 1924, 256, 8192));
	printf("MEM read w/ Cache Size Ablation (2048*128) = %zu\n", policySlotMIN(csr, simRowOrder, 2048, 128, 8192));
*/
	printf("MEM read w/ Cache Size Ablation (512*1024) = %zu\n", policySlotMIN(csr, simRowOrder, 512, 1024, 8192));
	printf("MEM read w/ Cache Size Ablation (1024*512) = %zu\n", policySlotMIN(csr, simRowOrder, 1024, 512, 8192));
	printf("MEM read w/ Cache Size Ablation (2048*256) = %zu\n", policySlotMIN(csr, simRowOrder, 2048, 256, 8192));
	printf("MEM read w/ Cache Size Ablation (4096*128) = %zu\n", policySlotMIN(csr, simRowOrder, 4096, 128, 8192));

	printf("MEM read w/ Cache Size Ablation (1024*1024) = %zu\n", policySlotMIN(csr, simRowOrder, 1024, 1024, 8192));
	printf("MEM read w/ Cache Size Ablation (2048*512) = %zu\n", policySlotMIN(csr, simRowOrder, 2048, 512, 8192));
	printf("MEM read w/ Cache Size Ablation (4096*256) = %zu\n", policySlotMIN(csr, simRowOrder, 4096, 256, 8192));
	printf("MEM read w/ Cache Size Ablation (8192*128) = %zu\n", policySlotMIN(csr, simRowOrder, 8192, 128, 8192));

	printf("MEM read w/ Cache Size Ablation (2048*1024) = %zu\n", policySlotMIN(csr, simRowOrder, 2048, 1024, 8192));
	printf("MEM read w/ Cache Size Ablation (4096*512) = %zu\n", policySlotMIN(csr, simRowOrder, 4096, 512, 8192));
	printf("MEM read w/ Cache Size Ablation (8192*256) = %zu\n", policySlotMIN(csr, simRowOrder, 8192, 256, 8192));
	printf("MEM read w/ Cache Size Ablation (16384*128) = %zu\n", policySlotMIN(csr, simRowOrder, 16384, 128, 8192));
#endif

	size_t cntCycle = simulateCycle(compact, csr, "");

	printf("Cycle = %zu\n", cntCycle);

#endif

	return 0;
}