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


	// TODO: Enforce that initMatLib is called only once (Singleton Manager?)
	//       and make _num_threads a static member of the class to prevent tamper
	static int 		 _num_threads  = 1;

	template <typename T = double> class NeoMatrix
	{
	    using VA_DATA = std::valarray<T>;
		using NEOMX = NeoMatrix<T>;
		
		private:
		
		size_t           _rows;
		size_t           _cols;
	    std::valarray<T> _data;
		
		public:

		void swap(NEOMX & x, NEOMX & y)
		{
			using std::swap;
			swap(x._rows, y._rows);
			swap(x._cols, y._cols);
			swap(x._data, y._data);
		}
		
		//   Constructors 
		
		NeoMatrix(const size_t & r = 0, const size_t & c = 0): _rows(r), _cols(c), _data(r * c)
		{
		}
		
		NeoMatrix(const size_t & r, const size_t & c, const VA_DATA & d): _rows(r), _cols(c), _data(d)
		{
		}
		
		NeoMatrix(const VA_DATA & d): _rows(1), _cols(d.size()), _data(d)
		{
		}
		
		// Initialize matrix data with a std::vector 
		NeoMatrix(const std::vector<std::vector<T>> & v):
		_rows(v.empty() ? 0 : v.size()), _cols(v.empty() ? 0 : v.front().size()), _data(_rows * _cols)
		{
			size_t k = 0;
			for(size_t i = 0; i < v.size(); ++i)
			{
				size_t size = v[i].size();
				assert(size == _cols);
				for(size_t j = 0; j < size; ++j) _data[k++] = v[i][j];
			}
		}
		
		// Initialize matrix data with an array of doubles calling default constructor 1st
		NeoMatrix(const size_t & r, const size_t & c, const T * d): NEOMX(r, c)
		{
			for(size_t i = 0; i < _rows * _cols; ++i) 
				_data[i] = d[i];
		}
		
		NeoMatrix(const size_t & r, const size_t & c, const T ** d): NEOMX(r, c)
		{
			size_t k = 0;
			for(size_t i = 0; i < _rows; ++i)
				for(size_t j = 0; j < _cols; ++j) 
					_data[k++] = d[i][j];
		}
		
		NeoMatrix(std::initializer_list<T> l): _rows(1), _cols(l.size()), _data(l)
		{
		}
		
		NeoMatrix(std::initializer_list<std::initializer_list<T>> l): NEOMX(l.size(), l.begin()->size())
		{
			size_t i = 0;
			for(const auto & j: l)
			{
				assert(j.size() == _cols);
				for(const auto & k: j) _data[i++] = k;
			}
		}
		
		//   MOVE Constructor
		
		NeoMatrix(NEOMX && m): NEOMX()
		{
			swap(*this, m);
		}
				
		//COPY Constructor
		
		NeoMatrix(const NEOMX & m): _rows(m._rows), _cols(m._cols), _data(m._data)
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
			NEOMX m(_rows, _cols, VA_DATA(n, _rows * _cols));
			swap(*this, m);
			return *this;
		}
		
		T & operator()(const size_t & i, const size_t & j)
		{
			return _data[i * _cols + j];
		}
		
		NEOMX operator+() const
		{
			return NEOMX(_rows, _cols, +_data);
		}
		
		NEOMX operator-() const
		{
			return NEOMX(_rows, _cols, -_data);
		}
		
		NeoMatrix<bool> operator!() const
		{
			return NeoMatrix<bool>(_rows, _cols, !_data);
		}
		
		NeoMatrix<bool> operator&&(const NEOMX & m) const
		{
			return NeoMatrix<bool>(_rows, _cols, _data && m._data);
		}
		
		NeoMatrix<bool> operator||(const NEOMX & m) const
		{
			return NeoMatrix<bool>(_rows, _cols, _data || m._data);
		}
		
		template <std::enable_if<std::is_integral<T>::value>* = nullptr>

		NEOMX operator%(const T & n) const
		{
			return NEOMX(_rows, _cols, _data % n);
		}
		
		template <std::enable_if<std::is_integral<T>::value>* = nullptr>
		NEOMX & operator%=(const T & n)
		{
			_data %= n;
			return *this;
		}
		
		template <std::enable_if<std::is_integral<T>::value>* = nullptr>
		NEOMX operator&(const T & n) const
		{
			return NEOMX(_rows, _cols, _data ^ n);
		}
		
		template <std::enable_if<std::is_integral<T>::value>* = nullptr>
		NEOMX & operator&=(const T & n)
		{
			_data &= n;
			return *this;
		}
		
		template <std::enable_if<std::is_integral<T>::value>* = nullptr>
		NEOMX operator^(const T & n) const
		{
			return NEOMX(_rows, _cols, _data ^ n);
		}
		
		template <std::enable_if<std::is_integral<T>::value>* = nullptr>
		NEOMX & operator^=(const T & n)
		{
			_data ^= n;
			return *this;
		}
		
		template <std::enable_if<std::is_integral<T>::value>* = nullptr>
		NEOMX operator|(const T & n) const
		{
			return NEOMX(_rows, _cols, _data | n);
		}
		
		template <std::enable_if<std::is_integral<T>::value>* = nullptr>
		NEOMX & operator|=(const T & n)
		{
			_data |= n;
			return *this;
		}
		
		NeoMatrix<bool> operator>(const NEOMX & m) const
		{
			assert((_rows == m._rows) && (_cols == m._cols));
			return NeoMatrix<bool>(_rows, _cols, _data > m._data);
		}
		
		NeoMatrix<bool> operator<(const NEOMX & m) const
		{
			assert((_rows == m._rows) && (_cols == m._cols));
			return NeoMatrix<bool>(_rows, _cols, _data < m._data);
		}
		
		NeoMatrix<bool> operator>=(const NEOMX & m) const
		{
			assert(_rows == m._rows && _cols == m._cols);
			return NeoMatrix<bool>(_rows, _cols, _data >= m._data);
		}
		
		NeoMatrix<bool> operator<=(const NEOMX & m) const 
		{
			assert(_rows == m._rows && _cols == m._cols);
			return NeoMatrix<bool>(_rows, _cols, _data <= m._data);
		}
		
		NeoMatrix<bool> operator==(const NEOMX & m) const
		{
			assert(_rows == m._rows && _cols == m._cols);
			return NeoMatrix<bool>(_rows, _cols, _data == m._data);
		}
		
		NeoMatrix<bool> operator!=(const NEOMX & m) const
		{
			assert(_rows == m._rows && _cols == m._cols);
			return NeoMatrix<bool>(_rows, _cols, _data != m._data);
		}
		
		template <std::enable_if<!std::is_same<T, bool>::value>* = nullptr>
		NEOMX operator++()
		{
			for(size_t i = 0; i < _rows * _cols; ++i) ++_data[i];
			return (*this);
		}
		
		template <std::enable_if<!std::is_same<T, bool>::value>* = nullptr>
		NEOMX operator++(int)
		{
			NEOMX temp = (*this);
			for(size_t i = 0; i < _rows * _cols; ++i) ++_data[i];
			return temp;
		}
		
		template <std::enable_if<!std::is_same<T, bool>::value>* = nullptr>
		NEOMX operator--()
		{
			for(size_t i = 0; i < _rows * _cols; ++i) --_data[i];
			return (*this);
		}
		
		template <std::enable_if<!std::is_same<T, bool>::value>* = nullptr>
		NEOMX operator--(int)
		{
			NEOMX temp = (*this);
			for(size_t i = 0; i < _rows * _cols; ++i) --_data[i];
			return temp;
		}
		
		NEOMX operator+(const T & n) const
		{
			return NEOMX(_rows, _cols, _data + n);
		}
		
		NEOMX & operator+=(const T & n)
		{
			_data += n;
			return *this;
		}
		
		NEOMX operator-(const T & n) const
		{
			return NEOMX(_rows, _cols, _data - n);
		}
		
		NEOMX & operator-=(const T & n)
		{
			_data -= n;
			return *this;
		}
		
		NEOMX operator*(const T & n) const
		{
			return NEOMX(_rows, _cols, _data * n);
		}
		
		NEOMX & operator*=(const T & n)
		{
			_data *= n;
			return *this;
		}
		
		NEOMX operator/(const T & n) const
		{
			return NEOMX(_rows, _cols, _data / n);
		}
		
		NEOMX & operator/=(const T & n)
		{
			_data /= n;
			return *this;
		}
		
		NEOMX operator+(const NEOMX & m) const
		{
			assert(m._rows == _rows && m._cols == _cols);
			return NEOMX(_rows, _cols, (_data + m._data));
		}
		
		NEOMX & operator+=(const NEOMX & m)
		{
			assert(m._rows == _rows && m._cols == _cols);
			_data += m._data;
			return *this;
		}
		
		NEOMX operator-(const NEOMX & m) const
		{
			assert(m._rows == _rows && m._cols == _cols);
			return NEOMX(_rows, _cols, (_data - m._data));
		}
		
		NEOMX & operator-=(const NEOMX & m)
		{
			assert(m._rows == _rows && m._cols == _cols);
			_data -= m._data;
			return *this;
		}
		
        // Matrix Multiplication 
		// 
		// multiplyBlock is the portion of one thread's work for the solution to be placed into
		//  param result
		static void multiplyBlock(VA_DATA & result, const int start_row, const int num_rows, const NEOMX & m1, const NEOMX & m2) 
		{
        	const int end_row = start_row + num_rows;
			for(auto i = start_row; i < end_row; ++i)
			{
				for(auto j = 0; j < m2._cols; ++j)
				{
					VA_DATA row = m1._data[std::slice(i * m1._cols, m1._cols, 1)];
					VA_DATA col = m2._data[std::slice(j, m2._rows, m2._cols)];
					result[i * m2._cols + j] = (row * col).sum();
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
			assert(_cols == m._rows);   // becomes a no-op if invalid operation

			VA_DATA result(_rows * m._cols);

			// container for our threads
			std::vector<std::thread> threads(_num_threads);

            // Calculate work load of each thread
            
			const int    block_size  = _rows/_num_threads ;
			const int    remaining   = _rows % _num_threads;
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

			return NEOMX(_rows, m._cols, result);
       
        } 



       ////////////////////////////////////////////////////
       // multiply - for single thread case

		NEOMX multiply(const NEOMX & m) const
		{
			// becomes a no-op if invalid operation
			assert(_cols == m._rows); 

			VA_DATA result(_rows * m._cols);
			for(size_t i = 0; i < _rows; ++i)
			{
				for(size_t j = 0; j < m._cols; ++j)
				{
					VA_DATA row = _data[std::slice(i * _cols, _cols, 1)];
					VA_DATA col = m._data[std::slice(j, m._rows, m._cols)];
					result[i * m._cols + j] = (row * col).sum();
				}
			}
			return NEOMX(_rows, m._cols, result);
		}

        ///////////////////////////////////////////////
        // the multiply operator decides whether to thread 
        // based on the threading flag _num_threads and
        // the number of rows to process
        ///////////////////////////////////////////////
		NEOMX operator*(const NEOMX & m) const
        {
			if ( (_num_threads > 1) && (_rows > _num_threads))
		      return(multiplyThreaded(m)) ;
		  	else
		   	 return( multiply(m)) ;
		}

		
		NEOMX & operator*=(const NEOMX & m)
		{
			assert(_cols == m._rows);
			NEOMX product = (*this) * m;
			swap(product, *this);
			return *this;
		}
		///////////////////////////////////////////
        // transpose() swap the rows and columns
        // of this matrix
		NEOMX transpose() const
		{
			NEOMX m(_cols, _rows);
			for(size_t i = 0; i < _rows; ++i)
				m._data[std::slice(i, _cols, _rows)] = _data[std::slice(i * _cols, _cols, 1)];
			return std::move(m);
		}
		
		
		// Methods
		

		void initialize_random() 
		{
    		std::random_device rd;
    		std::mt19937 mt(rd());
    		std::uniform_real_distribution<double> dist(0.0, 9.9);
    		auto random = std::bind(dist, mt);
    		for (int i = 0; i < _rows*_cols; ++i) 
			{
        		_data[i] = random();
      		}
    	}

		size_t getNumRows() const
		{
			return _rows;
		}
		
		size_t getNumCols() const
		{
			return _cols;
		}
		
		
		void fill(const T & n)
		{
			_data = n;
		}
		
		void clear()
		{
			_rows = 0;
			_cols = 0;
			_data.resize(0);
		}
		
		NEOMX apply(T func(const T &))
		{
			return NEOMX(_rows, _cols, _data.apply(func));
		}
		
		bool isEmpty() const
		{
			return !(_rows && _cols);
		}
		
		bool isEqual(const NEOMX & m) const
		{
			return (_rows == m._rows) && (_cols == m._cols) && !(_data == m._data).sum();
		}
		
		bool isSizeOf(const NEOMX & m) const
		{
			return (_rows == m._rows) && (_cols == m._cols);
		}
		
		
	//  Print Utilities
		
		
		void print(std::ostream & os = std::cout) const
		{
			for(size_t i = 0; i < _rows * _cols; ++i)
			{

				os << ((i  % _cols) ? "" : "|\t");
				os << _data[i] << "  ";
				os << (((i + 1) % _cols) ? "  " : "\t|\n");
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


	// TODO: Enforce that initMatLib is called only once (Singleton Manager?)
	//       and make _num_threads a static member of the class to prevent tamper
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
