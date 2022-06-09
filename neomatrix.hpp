// FILE:	neomatrix.hpp
// AUTHOR:	Amber Rogowicz
// DATE:	July 2018
//
//------------------------------------------------------------------------------------------
//	The Interface †o NeoLib:
//
//		Namespace of NeoLib		: "using namespace neolib"
//		Initializing threading	: initMatLib( int maxThreadCount) ; Runs the library operations threaded with maxThreadCount
//																	or single threaded if maxThreadCount <= 1
//																	NOTE: can be called anytime throughout execution
//		NeoMatrix class			: "Matrix" declaration/instantiation
//		Initialization or fill	: initialize_random() Method random init data of a Matrix distribution range 0.0-9.9
//                 				: Matrix A[4][2]  = 1.4; - Inits all data elements to 1.4
//      Init Constructor		: Matrix(const std::vector<std::vector<T>> & v): Init matrix data with a std::vector 
//      initializer_list		: Matrix A =  { {2.0 , 3.0, 4.0 },
//												{ 1.3, 4.7, 1.3 }} ; inits A[2][3]
//		Multiply 				: "*" operator of 2 or more instantiated Matrix 
//		         				: "*=" operator of an instantiated Matrix 
//		Transpose				: transpose() method of a Matrix
//		Error Handling			: Currently exits application with an assert() failure
//
//       Defines a template NeoMatrix class created using the STL std::valarray 
// 
//--------------------------------------------------------------------------------
// 
// NeoMatrix
//       Is a Matrix class created using the STL std::valarray 
//		 which implements methods:
//			Matrix Multiplication  - ResultMatrix(MxP) = MatrixA(MxN) * MatrixB{NxP)
//          Matrix Transpose       - Matrix(NxM)       = Matrix(MxN)
//       STL std::threads are used to speed up the Matrix Multiplication
//
//--------------------------------------------------------------------------------

// neomatrix.hpp

#ifndef NEOMATRIX_HPP
#define NEOMATRIX_HPP


#include <cstdlib>
#include <cassert>
#include <utility>
#include <valarray>
#include <vector>
#include <iostream>
#include <type_traits>
#include <initializer_list>
#include <random>
#include <thread>
#include <unistd.h>

#define MATLIB_ALIGN(a) __attribute__((aligned(a)))
#define MATLIB_ENSURE_INLINE __attribute__((always_inline))

namespace matlib
{


	static int _num_threads = 1;
	static int _finished_threads = 0;

	template <typename T = double> class NeoMatrix
	{
	    using VA_DATA = std::valarray<T>;
		using NEOMX = NeoMatrix<T>;
		
		private:
		
		size_t           rows;
		size_t           cols;
	    std::valarray<T> data;
		
		public:

		void swap(NEOMX & x, NEOMX & y)
		{
			using std::swap;
			swap(x.rows, y.rows);
			swap(x.cols, y.cols);
			swap(x.data, y.data);
		}
		
		//   Constructors 
		
		NeoMatrix(const size_t & r = 0, const size_t & c = 0): rows(r), cols(c), data(r * c)
		{
		}
		
		NeoMatrix(const size_t & r, const size_t & c, const VA_DATA & d): rows(r), cols(c), data(d)
		{
		}
		
		NeoMatrix(const VA_DATA & d): rows(1), cols(d.size()), data(d)
		{
		}
		
		// Initialize matrix data with a std::vector 
		NeoMatrix(const std::vector<std::vector<T>> & v):
		rows(v.empty() ? 0 : v.size()), cols(v.empty() ? 0 : v.front().size()), data(rows * cols)
		{
			size_t k = 0;
			for(size_t i = 0; i < v.size(); ++i)
			{
				size_t size = v[i].size();
				assert(size == cols);
				for(size_t j = 0; j < size; ++j) data[k++] = v[i][j];
			}
		}
		
		// Initialize matrix data with an array of doubles calling default constructor 1st
		NeoMatrix(const size_t & r, const size_t & c, const T * d): NEOMX(r, c)
		{
			for(size_t i = 0; i < rows * cols; ++i) 
				data[i] = d[i];
		}
		
		NeoMatrix(const size_t & r, const size_t & c, const T ** d): NEOMX(r, c)
		{
			size_t k = 0;
			for(size_t i = 0; i < rows; ++i)
				for(size_t j = 0; j < cols; ++j) 
					data[k++] = d[i][j];
		}
		
		NeoMatrix(std::initializer_list<T> l): rows(1), cols(l.size()), data(l)
		{
		}
		
		NeoMatrix(std::initializer_list<std::initializer_list<T>> l): NEOMX(l.size(), l.begin()->size())
		{
			size_t i = 0;
			for(const auto & j: l)
			{
				assert(j.size() == cols);
				for(const auto & k: j) data[i++] = k;
			}
		}
		
		//   MOVE Constructor
		
		NeoMatrix(NEOMX && m): NEOMX()
		{
			swap(*this, m);
		}
				
		//COPY Constructor
		
		NeoMatrix(const NEOMX & m): rows(m.rows), cols(m.cols), data(m.data)
		{
		}
		
		//  ASSIGNMENT Constructor
		
		NEOMX & operator=(NEOMX m)
		{
			swap(*this, m);
			return *this;
		}

		NEOMX & operator=(NEOMX & m)
		{
			swap(*this, m);
			return *this;
		}
		
		
		// Operators
		
		NEOMX & operator=(const T & n)
		{
			NEOMX m(rows, cols, VA_DATA(n, rows * cols));
			swap(*this, m);
			return *this;
		}
		
		T & operator()(const size_t & i, const size_t & j)
		{
			return data[i * cols + j];
		}
		
		NEOMX operator+() const
		{
			return NEOMX(rows, cols, +data);
		}
		
		NEOMX operator-() const
		{
			return NEOMX(rows, cols, -data);
		}
		
		NeoMatrix<bool> operator!() const
		{
			return NeoMatrix<bool>(rows, cols, !data);
		}
		
		NeoMatrix<bool> operator&&(const NEOMX & m) const
		{
			return NeoMatrix<bool>(rows, cols, data && m.data);
		}
		
		NeoMatrix<bool> operator||(const NEOMX & m) const
		{
			return NeoMatrix<bool>(rows, cols, data || m.data);
		}
		
		template <std::enable_if<std::is_integral<T>::value>* = nullptr>

		NEOMX operator%(const T & n) const
		{
			return NEOMX(rows, cols, data % n);
		}
		
		template <std::enable_if<std::is_integral<T>::value>* = nullptr>
		NEOMX & operator%=(const T & n)
		{
			data %= n;
			return *this;
		}
		
		template <std::enable_if<std::is_integral<T>::value>* = nullptr>
		NEOMX operator&(const T & n) const
		{
			return NEOMX(rows, cols, data ^ n);
		}
		
		template <std::enable_if<std::is_integral<T>::value>* = nullptr>
		NEOMX & operator&=(const T & n)
		{
			data &= n;
			return *this;
		}
		
		template <std::enable_if<std::is_integral<T>::value>* = nullptr>
		NEOMX operator^(const T & n) const
		{
			return NEOMX(rows, cols, data ^ n);
		}
		
		template <std::enable_if<std::is_integral<T>::value>* = nullptr>
		NEOMX & operator^=(const T & n)
		{
			data ^= n;
			return *this;
		}
		
		template <std::enable_if<std::is_integral<T>::value>* = nullptr>
		NEOMX operator|(const T & n) const
		{
			return NEOMX(rows, cols, data | n);
		}
		
		template <std::enable_if<std::is_integral<T>::value>* = nullptr>
		NEOMX & operator|=(const T & n)
		{
			data |= n;
			return *this;
		}
		
		NeoMatrix<bool> operator>(const NEOMX & m) const
		{
			assert(rows == m.rows && cols == m.cols);
			return NeoMatrix<bool>(rows, cols, data > m.data);
		}
		
		NeoMatrix<bool> operator<(const NEOMX & m) const
		{
			assert(rows == m.rows && cols == m.cols);
			return NeoMatrix<bool>(rows, cols, data < m.data);
		}
		
		NeoMatrix<bool> operator>=(const NEOMX & m) const
		{
			assert(rows == m.rows && cols == m.cols);
			return NeoMatrix<bool>(rows, cols, data >= m.data);
		}
		
		NeoMatrix<bool> operator<=(const NEOMX & m) const 
		{
			assert(rows == m.rows && cols == m.cols);
			return NeoMatrix<bool>(rows, cols, data <= m.data);
		}
		
		NeoMatrix<bool> operator==(const NEOMX & m) const
		{
			assert(rows == m.rows && cols == m.cols);
			return NeoMatrix<bool>(rows, cols, data == m.data);
		}
		
		NeoMatrix<bool> operator!=(const NEOMX & m) const
		{
			assert(rows == m.rows && cols == m.cols);
			return NeoMatrix<bool>(rows, cols, data != m.data);
		}
		
		template <std::enable_if<!std::is_same<T, bool>::value>* = nullptr>
		NEOMX operator++()
		{
			for(size_t i = 0; i < rows * cols; ++i) ++data[i];
			return (*this);
		}
		
		template <std::enable_if<!std::is_same<T, bool>::value>* = nullptr>
		NEOMX operator++(int)
		{
			NEOMX temp = (*this);
			for(size_t i = 0; i < rows * cols; ++i) ++data[i];
			return temp;
		}
		
		template <std::enable_if<!std::is_same<T, bool>::value>* = nullptr>
		NEOMX operator--()
		{
			for(size_t i = 0; i < rows * cols; ++i) --data[i];
			return (*this);
		}
		
		template <std::enable_if<!std::is_same<T, bool>::value>* = nullptr>
		NEOMX operator--(int)
		{
			NEOMX temp = (*this);
			for(size_t i = 0; i < rows * cols; ++i) --data[i];
			return temp;
		}
		
		NEOMX operator+(const T & n) const
		{
			return NEOMX(rows, cols, data + n);
		}
		
		NEOMX & operator+=(const T & n)
		{
			data += n;
			return *this;
		}
		
		NEOMX operator-(const T & n) const
		{
			return NEOMX(rows, cols, data - n);
		}
		
		NEOMX & operator-=(const T & n)
		{
			data -= n;
			return *this;
		}
		
		NEOMX operator*(const T & n) const
		{
			return NEOMX(rows, cols, data * n);
		}
		
		NEOMX & operator*=(const T & n)
		{
			data *= n;
			return *this;
		}
		
		NEOMX operator/(const T & n) const
		{
			return NEOMX(rows, cols, data / n);
		}
		
		NEOMX & operator/=(const T & n)
		{
			data /= n;
			return *this;
		}
		
		NEOMX operator+(const NEOMX & m) const
		{
			assert(m.rows == rows && m.cols == cols);
			return NEOMX(rows, cols, (data + m.data));
		}
		
		NEOMX & operator+=(const NEOMX & m)
		{
			assert(m.rows == rows && m.cols == cols);
			data += m.data;
			return *this;
		}
		
		NEOMX operator-(const NEOMX & m) const
		{
			assert(m.rows == rows && m.cols == cols);
			return NEOMX(rows, cols, (data - m.data));
		}
		
		NEOMX & operator-=(const NEOMX & m)
		{
			assert(m.rows == rows && m.cols == cols);
			data -= m.data;
			return *this;
		}
		
        // Matrix Multiplication 
		// 
		// multiplyBlock is the portion of one thread's work for the solution to be placed into
		//  param result
		static void multiplyBlock(VA_DATA & result, 
								  const int start_row, const int num_rows, 
								  const NEOMX & m1,    const NEOMX & m2) 
		{
        	const int end_row = start_row + num_rows;
			for(auto i = start_row; i < end_row; ++i)
			{
				for(auto j = 0; j < m2.cols; ++j)
				{
					VA_DATA row = m1.data[std::slice(i * m1.cols, m1.cols, 1)];
					VA_DATA col = m2.data[std::slice(j, m2.rows, m2.cols)];
					result[i * m2.cols + j] = (row * col).sum();
				}
			}
			return ;
		}


		////////////////////////////////////////////////////////////////////////////////
        // overlapped processing of the Matrix Multiply using std::thread ( built on posix)
        // Compute the number of rows for each thread to handle computing for the 
        // result matrix and spawn threads with work method "multiplyBlock()"

		NEOMX multiplyThreaded(const NEOMX & m) const
		{
			assert(cols == m.rows);   // becomes a no-op if invalid operation

			VA_DATA result(rows * m.cols);

			// container for our threads
			std::vector<std::thread> threads(_num_threads);

            // Calculate work load of each thread
            
			const int    block_size  = rows/_num_threads ;
			const int    remaining   = rows % _num_threads;
            const int    last_thread = _num_threads - 1;

			int          start_row = 0;
  			for (int i = 0; i < last_thread; ++i) 
			{
                // Now hand the work load of block_size (number of rows to process) to 
                // the threads to handle in the static method multiplyBlock() passing
				// references to the valarray data and the this ptr
				threads[i] = (std::thread(& NEOMX::multiplyBlock, 
							std::ref(result), start_row, block_size, std::ref(*this), std::ref(m)));

                // increment the starting index for the next block of work
				start_row += block_size;
  			}
			// take care of the last block and tack on any remaining rows left over from our thread division
			threads[last_thread] = (std::thread(& NEOMX::multiplyBlock, std::ref(result), 
								start_row, block_size + remaining, std::ref(*this), std::ref(m)));

			// Regroup after threads finish work
			// for(auto &t : threads)
			for(int thread_idx = 0;  thread_idx < _num_threads; thread_idx++ )
			{
				threads[thread_idx].join();
  			}

			return NEOMX(rows, m.cols, result);
       
        } 



       ////////////////////////////////////////////////////
       // multiply - for single thread case

		NEOMX multiply(const NEOMX & m) const
		{
			// becomes a no-op if invalid operation
			assert(cols == m.rows); 

			VA_DATA result(rows * m.cols);
			for(size_t i = 0; i < rows; ++i)
			{
				for(size_t j = 0; j < m.cols; ++j)
				{
					VA_DATA row = data[std::slice(i * cols, cols, 1)];
					VA_DATA col = m.data[std::slice(j, m.rows, m.cols)];
					// result[i * m.cols + j] = (row * col).sum();
					result[i * m.cols + j] = row * col;
				}
			}
			return NEOMX(rows, m.cols, result);
		}

        ///////////////////////////////////////////////
        // the multiply operator decides whether to thread 
        // based on the threading flag _num_threads and
        // the number of rows to process
        ///////////////////////////////////////////////
		NEOMX operator*(const NEOMX & m) const
        {
			if ( (_num_threads > 1) && (rows > _num_threads))
		      return(multiplyThreaded(m)) ;
		  	else
		   	 return( multiply(m)) ;
		}

		
		NEOMX & operator*=(const NEOMX & m)
		{
			assert(cols == m.rows);
			NEOMX product = (*this) * m;
			swap(product, *this);
			return *this;
		}
		///////////////////////////////////////////
        // transpose() swap the rows and columns
        // of this matrix
		NEOMX transpose() const
		{
			NEOMX m(cols, rows);
			for(size_t i = 0; i < rows; ++i)
				m.data[std::slice(i, cols, rows)] = data[std::slice(i * cols, cols, 1)];
			return std::move(m);
		}
		
		
		// Methods
		
		void initialize_random() 
		{
    		std::random_device rd;
    		std::mt19937 mt(rd());
    		std::uniform_real_distribution<double> dist(-1.0, 1.0);
    		auto random = std::bind(dist, mt);
    		for (int i = 0; i < rows*cols; ++i) 
			{
        		data[i] = random();
      		}
    	}

		size_t getNumRows() const
		{
			return rows;
		}
		
		size_t getNumCols() const
		{
			return cols;
		}
		
		
		void fill(const T & n)
		{
			data = n;
		}
		
		void clear()
		{
			rows = 0;
			cols = 0;
			data.resize(0);
		}
		
		NEOMX apply(T func(const T &))
		{
			return NEOMX(rows, cols, data.apply(func));
		}
		
		bool isEmpty() const
		{
			return !(rows && cols);
		}
		
		bool isEqual(const NEOMX & m) const
		{
			return (rows == m.rows) && (cols == m.cols) && !(data == m.data).sum();
		}
		
		bool isSizeOf(const NEOMX & m) const
		{
			return (rows == m.rows) && (cols == m.cols);
		}
		
		
	//  Print Utilities
		
		
		void print(std::ostream & os = std::cout) const
		{
			for(size_t i = 0; i < rows * cols; ++i)
			{

				os << ((i  % cols) ? "" : "|\t");
				os << data[i] << "  ";
				os << (((i + 1) % cols) ? "  " : "\t|\n");
			}
			os <<  std::endl << std::endl;
			os.flush();
		}
	
	}; // end NeoMatrix class definition

	template <typename T> std::ostream & operator<<(std::ostream & os, const NeoMatrix<T> & matrix)
	{
		matrix.print(os);
		return os;
	}

	static void initMatLib( const int thread_count)
    {
        // if thread_count > 1 we are multithreading our matrix multiply 
		_num_threads = thread_count ;
		// TODO: set matrix multiply function instead of testing threaded flag each time
    }

	template class NeoMatrix<double>;
	template class NeoMatrix<float>;
	template class NeoMatrix<int>;
	template class NeoMatrix<long>;
	template class NeoMatrix<short>;
	template class NeoMatrix<unsigned>;
	template class NeoMatrix<size_t>;
	template class NeoMatrix<bool>;
	
	
	using Matrix = NeoMatrix<>;

}  // end namespace matlib

#endif // #ifndef NEOMATRIX_HPP
