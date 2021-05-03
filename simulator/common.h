#pragma once

#include <cstdint>
#include <vector>


typedef uint32_t index_t;
typedef double value_t;

#pragma pack(push, 1)
struct CSRElement
{
	index_t idx;
	value_t val;
};
#pragma pack(pop)

struct COOElement
{
	// uint32_t unused;

	index_t row, col;
	value_t val;

	COOElement() {}
	constexpr COOElement(index_t row, index_t col, value_t val) : row(row), col(col), val(val) {}
	constexpr COOElement(index_t row, CSRElement csr) : row(row), col(csr.idx), val(csr.val) {}

	bool operator<(const COOElement &rhs) const
	{
		return row == rhs.row ? col < rhs.col : row < rhs.row;
	}
};



//const size_t S = sizeof(COOElement);

struct CSRMatrix
{
	std::vector<size_t>     pos;
	std::vector<CSRElement> data;
	//std::vector<index_t> idx;
	//std::vector<value_t> val;

	size_t NRow() const { return pos.size() - 1; }
};

typedef std::vector<COOElement> COOMatrix;
//typedef std::vector<std::vector<COOElement>> CompactCOOMatrix;

struct CompactCOOMatrix
{
	std::vector<size_t> pos;
	std::vector<COOElement> data;
};

template<typename T>
T alignTo(T input, T align) {
	return (input + align - 1) / align * align;
}