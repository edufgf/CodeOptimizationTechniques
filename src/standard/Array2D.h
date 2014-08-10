#include <immintrin.h>

template<class E> class Array2D
{
public:

	    Array2D() { 
	    	construct(0, 0, 0); 
	    }
	    
	    Array2D(int dim1, int dim2) { 
	    	construct(dim1, dim2, 0); 
	    }
	    
	    Array2D(int dim1, int dim2, int align) { 
	    	construct(dim1, dim2, align); 
	    }
	    
	    Array2D(int dim1, int dim2, int align, const E &val) { 
	    	construct(dim1, dim2, align); fill(val); 
	    }
		
	    // Returns dimension 1 length.
	    int dim1() const { return m_dim1; }
	    
	    // Returns dimension 2 length.
	    int dim2() const { return m_dim2; }
	    
	    // Returns allocated (dim1*dim2) length.
	    int size() const { return m_size; }
	    
	    void build(int dim1, int dim2, int align, const E &val) {
	    	if (m_pStart!=NULL)
	    		deconstruct();	
	    	construct(dim1, dim2, align);
	    	fill(val);
	    }
	    
	    
	    void build(int dim1, int dim2, int align) {
	    	if (m_pStart!=NULL)
	    		deconstruct();	
	    	construct(dim1, dim2, align);
	    }
	    
	    // Fill the matrix with val.
		void fill(const E &val) {
			for (int i = 0; i < m_size; ++i)
				m_pStart[i] = val;	
		}
		
		// Set matrix position at (i, j) with val.
		void set(int i, int j, const E &val) {
			//OGDF_ASSERT(0 <= i && i <= m_dim1 && 0 <= j && j <= m_dim2);
			m_pStart[i*m_dim2+j] = val;
		}
		
		// Set matrix position at index with val.
		void set(int index, const E &val) {
			//OGDF_ASSERT(0 <= index && index < m_size);
			m_pStart[index] = val;
		}
		
		int getCollumnSize() const {
			return m_dim2;
		}
		
		int getSize() const {
			return m_size;
		}
		
		E* getAddress(int index) const {
			return &m_pStart[index];
		}
		
		// Return a reference to the element with index (i, j).
	    E &operator()(int i, int j) const {
			//OGDF_ASSERT(0 <= i && i <= m_dim1 && 0 <= j && j <= m_dim2);
			return m_pStart[i*m_dim2+j];
		}
		
		// Return a reference to the element at index.
		E &operator()(int index) const {
			//OGDF_ASSERT(0 <= index && index < m_size);
			return m_pStart[index];
		}
		
		Array2D<E> &operator=(const Array2D<E> &array2) {
			copy(array2);
			return *this;
		}
		
		void copy(const Array2D<E> &array2){
			if (!(array2.m_dim1 == m_dim1 && array2.m_dim2 == m_dim2 && array2.m_align == m_align)){
				deconstruct();
				construct(array2.m_dim1, array2.m_dim2, m_align);
			}

			if (m_pStart != 0) {
				E *pSrc  = array2.m_pStart;
				for (int i = 0; i < m_size; ++i)
					m_pStart[i] = pSrc[i];	
			}
		}
		
	    ~Array2D() { 
	    	deconstruct();
		}
	    
private:

		E   *m_pStart; 	// Start of the array (address of A[0,0]).
		int  m_dim1;
		int  m_dim2;
		int  m_size;	// m_dim1 * m_dim 2
		int  m_align;

		// Allocates a matrix as an array of length m_size.
		void construct(int dim1, int dim2, int align) {
		    m_dim1 = dim1;
		    m_dim2 = dim2;
		    m_size = dim1*dim2;
		    m_align = align;
		    
			if (m_size == 0) {
				m_pStart = NULL;
			} else {	
				if (!align)
					m_pStart = (E*) malloc (m_size * sizeof(E));
				else
					m_pStart = (E*) _mm_malloc (m_size * sizeof(E), 32);
				//if (m_pStart == NULL) /*Error*/
					//OGDF_THROW(InsufficientMemoryException);	
			}
		}
		
		void deconstruct() {
			if (m_pStart!=NULL)
				free(m_pStart);
		}
			   	
};

