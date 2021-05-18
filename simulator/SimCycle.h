#pragma once

#include "common.h"
#include <string>
#include <functional>
#include <deque>
#include <stdexcept>
#include <cassert>
#include <list>

template<typename T>
struct PortWrite
{
	std::function<size_t()> getAvailableSlot;
	std::function<void(std::vector<T>)> write;

	bool isWritable() const
	{
		return getAvailableSlot() > 0;
	}
};

template<typename T>
struct PortRead
{
	std::function<size_t()> getAvailableSlot;
	std::function<std::vector<T>(size_t)> read;
	std::function<std::vector<T>(size_t)> peek;
	std::function<void(size_t)> consume;

	bool isReadable() const
	{
		return getAvailableSlot() > 0;
	}
};

struct PortAddr
{
	std::function<void(size_t, bool, long)> access;
	std::function<bool()> poll;		// return true if i/o is done
};

class DRAMStats
{
public:
	virtual std::pair<size_t, size_t> getDRAMStats() = 0;	// { Read, Write }
};

class SRAMStats
{
public:
	virtual std::tuple<std::string, size_t, size_t, size_t, size_t> getSRAMStats() = 0;	// { name, size, width, read, write }
};

class Module
{
public:
	Module()
	{
		listModules.push_back(this);
		iterList = std::prev(listModules.end());
	}
	virtual ~Module()
	{
		listModules.erase(iterList);
	}

	virtual void clockUpdate() = 0;
	virtual void clockApply() = 0;
	virtual void printStats() {}

public:
	static void updateAll()
	{
		for (Module *mod : listModules)
			mod->clockUpdate();
		for (Module *mod : listModules)
		{
			mod->clockApply();
			mod->cntCycle++;
		}
	}
	static void printStatsAll()
	{
		for (Module *mod : listModules)
			mod->printStats();
	}

	template<typename T>
	static void foreach(std::function<void(T &)> func)
	{
		for (Module *mod : listModules)
		{
			T *modT = dynamic_cast<T *>(mod);
			if (modT)
				func(*modT);
		}
	}

protected:
	size_t cntCycle = 0;

private:
	std::list<Module *>::iterator iterList;

private:
	static std::list<Module *> listModules;
};


template<typename T>
class FIFO : public Module, public SRAMStats
{
public:
	FIFO(std::string name, size_t capacity, size_t rwidth, size_t wwidth = 0, size_t rptwidth = 0) :
		name(name),
		capacity(capacity), rwidth(rwidth), wwidth(wwidth), rptwidth(rptwidth), size(0)
	{
		if (wwidth == 0)
			this->wwidth = rwidth;
		if (rptwidth == 0)
			this->rptwidth = std::max(this->rwidth, this->wwidth);
		assert(this->rptwidth >= this->rwidth || this->rptwidth >= this->wwidth);
	}

	PortRead<T> getReadPort()
	{
		PortRead<T> port;
		port.getAvailableSlot = [this]()
		{
			return size;
		};
		port.read = [this](size_t amount)
		{
			if (dirtyRead)
				throw std::runtime_error("Multiple Read in one cycle");
			if (amount > size)
				throw std::runtime_error("FIFO underflow");
			if (amount > rwidth)
				throw std::runtime_error("Read size is greater than port width");

			dirtyRead = true;

			std::vector<T> ret;
			for (size_t i = 0; i < amount; i++)
			{
				ret.push_back(std::move(data.front()));
				data.pop_front();
			}

			sizeRead += rwidth * sizeof(T);

			return ret;
		};
		port.peek = [this](size_t amount)
		{
			if (size < amount)
				throw std::runtime_error("FIFO underflow");
			std::vector<T> ret;
			for (size_t i = 0; i < amount; i++)
			{
				ret.push_back(data[i]);
			}
			return ret;
		};
		port.consume = [this](size_t amount)
		{
			if (dirtyRead)
				throw std::runtime_error("Multiple Read in one cycle");
			dirtyRead = true;
			for (size_t i = 0; i < amount; i++)
				data.pop_front();

			sizeRead += rwidth * sizeof(T);
		};

		return port;
	}

	PortWrite<T> getWritePort()
	{
		PortWrite<T> port;

		port.getAvailableSlot = [this]()
		{
			return capacity - size;
		};
		port.write = [this](std::vector<T> elements)
		{
			if (dirtyWrite)
				throw std::runtime_error("Multiple Write in one cycle");
			if (capacity - size < elements.size())
				throw std::runtime_error("FIFO overflow");
			if (elements.size() > wwidth)
				throw std::runtime_error("Write size is greater than port width");

			dirtyWrite = true;

			for (auto &&e : elements)
				data.push_back(std::move(e));

			sizeWrite += wwidth * sizeof(T);
		};

		return port;
	}

	virtual void clockUpdate() override {}
	virtual void clockApply() override
	{
		dirtyRead = false;
		dirtyWrite = false;
		size = data.size();
	}

	virtual std::tuple<std::string, size_t, size_t, size_t, size_t> getSRAMStats() override
	{
		return { "FIFO-" + name, capacity * sizeof(T), rptwidth * sizeof(T), sizeRead, sizeWrite };
	}


protected:
	std::string name;
	size_t capacity;
	size_t rwidth, wwidth, rptwidth;	// in number of T
	size_t size;	// only updates during clockApply
	bool dirtyRead = false, dirtyWrite = false;
	std::deque<T> data;

	size_t sizeRead = 0;
	size_t sizeWrite = 0;
};