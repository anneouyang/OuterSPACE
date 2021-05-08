#pragma once

#include <vector>
#include <list>

class SimCacheLine
{
public:
	SimCacheLine(size_t numGroup) : lastAccess(numGroup) {}

	bool access(uintptr_t addr)
	{
		numRef++;

		bool miss = true;

		for (auto iter = lastAccess.begin(); iter != lastAccess.end(); iter++)
		{
			if (iter->valid && iter->addr == addr)
			{
				lastAccess.erase(iter);
				miss = false;
				break;
			}
		}

		if (miss)
		{
			lastAccess.pop_back();
			numMiss++;
		}

		lastAccess.push_front(CacheBlock{ addr, true });

		return miss;
	}

	size_t getNumMiss() const { return numMiss; }
	size_t getNumRef() const { return numRef; }

protected:
	struct CacheBlock
	{
		uintptr_t addr;
		bool valid = false;
	};
	std::list<CacheBlock> lastAccess;

	size_t numRef = 0;
	size_t numMiss = 0;
};


class SimCache
{
public:
	SimCache(size_t numGroup, size_t bitRowId, size_t bitByteSel) : lines(size_t(1) << bitRowId, SimCacheLine(numGroup)), bitRowId(bitRowId), bitByteSel(bitByteSel) {}

	// return true if miss
	bool access(uintptr_t addr)	
	{	
		size_t lineId = (addr >> bitByteSel) & ((1 << bitRowId) - 1);
		addr &= ~((1 << (bitByteSel + bitRowId)) - 1);
		return lines[lineId].access(getLineAddr(addr));
		// printf("REF %zu %p\n", lineId, addr);
	}

	size_t getNumMiss() const
	{
		size_t sum = 0;
		for (auto &&line : lines)
			sum += line.getNumMiss();
		return sum;
	}

	size_t getNumRef() const
	{
		size_t sum = 0;
		for (auto &&line : lines)
			sum += line.getNumRef();
		return sum;
	}

	uintptr_t getLineAddr(uintptr_t addr) const {
		return addr & ~((1 << bitByteSel) - 1);
	}

protected:
	size_t bitRowId, bitByteSel;
	std::vector<SimCacheLine> lines;
};