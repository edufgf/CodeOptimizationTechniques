#include <string.h>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <unistd.h>  
#include <iomanip>
#include "Array2D.h"
#include <math.h>

using namespace std;

bool is_number(const std::string& s) {
    return (strspn( s.c_str(), "0123456789") == s.size());
}

const int g_programCount = 19;	 // Matrix multiplication program versions.
const double g_EPS = 1;	 // EPS used when checking answers.
const int g_outputPrecision = 3; // Timing output precision.
const int g_outputWidth = 25;	 // Output Formatting.
int g_dataType;
string g_dataTypeStr;
string g_optimizationLevelStr;

const char* program_name[] = {
"standardMult", 
"optimizedMult", 
"optimizedMult2", 
"optimizedMultParallel", 
"optimizedMultParallel2", 
"strassenMult", 
"strassenMult2", 
"strassenMultParallel", 
"strassenMultParallel2", 
"recursiveMult", 
"recursiveMult2", 
"recursiveMultParallel", 
"recursiveMultParallel2", 
"openBlas", 
"openBlasParallel", 
"standardOpenCL", 
"optimizedOpenCL", 
"AMDopenCL", 
"clMath"};

// Handle command line parameters
int handleParameters(int argc, char *argv[]) {
	if (argc == 1) {
		cout << "Execute again with command line parameters!" << endl;
		cout << "\"./program -help\" for more information" << endl << endl;
		return 0;
	}
	
	string aux = argv[1];
	if ( aux == "-help" || aux == "help") {
		//Help info	
		cout << endl;
		cout << "Execute \"./test-mult M K N C\", where M, K and N are positive integers." << endl;
		cout << "Multiply matrices A[M,K] * B[K,N] = C[M,N]." << endl;
		cout << "M, K and N can be distinct integers." << endl;
		cout << "If any of the integers are not power of two, the strassen algorithm won't be executed." << endl;
		cout << "For optimal performance of the recursive algorithm, integers should be power of two." << endl;
		cout << endl;
		cout << "Algorithms executed: " << endl;
		cout << "Standard Multiplication." << endl;
		cout << "Standard Multiplication Optimized - Loop unrolling x5 + cache blocking memory." << endl;
		cout << "Strassen Multiplication - Some memory saving improvement, writing directly to output matrix." << endl;
		cout << "Strassen Multiplication Parallel." << endl;
		cout << "Recursive Matrix Multiplication - Z-order memory layout + column-major order (matrix B)." << endl;
		cout << "Recursive Matrix Multiplication Paralllel - Task scheduler." << endl;
		cout << endl;
		return 0;
	}
	
	int ret = 1;
	aux = argv[1];
	if (!is_number(aux))
		ret = 0;
	aux = argv[2];
	if (!is_number(aux))
		ret = 0;
	aux = argv[3];
	if (!is_number(aux))
		ret = 0;	
	if (ret == 0) {
		cout << "Invalid parameters!" << endl;
		cout << "\"./test-mult -help\" for more information." << endl << endl;	
		return 0;
	}	
	return 1;
}

// Function that asks user which algorithm to skip
void requestSkip(int* skip) {
	char op;
	cout << endl;
	cout << "Choose \"y\" (yes) or \"n\" (no) to skip functions." << endl << endl;
	for (int i = 0; i < g_programCount; ++i) {
		// Only asks for algorithms not skipped
		if (!skip[i]) {
			cout << "Skip " << program_name[i] << " ?" << endl;
			cin >> op;
			if (op=='y')
				skip[i] = 1;
		}	
	}
	return;
}

void getConsole(int * readConsole) {
	char op;
	cout << endl << "Read from console? (y/n)" << endl;
	cin >> op;
	if (op=='n')
		*readConsole = 0;
	else 
		*readConsole = 1;
	return;
}

void getRange(int * randomRangePos, int * onlyPositives) {
	cout << "Enter the range integer which the numbers should be generated [-range,+range]." << endl;
	cout << "Eg. range integer = 10, numbers generated within [-10,+10]." << endl;
	cin >> *randomRangePos;
	cout << endl << "Type 1 if you want only positive numbers or type anything else to keep the negative part." << endl;
	cin >> *onlyPositives;
}

void getCacheSize(int * cacheBlockSize){
	while(1) {
		printf("\nChoose the cache block size (must be integer)\n");
		string aux;
		cin >> aux;
		if (*cacheBlockSize < 0 || !is_number(aux) )
			printf("Invalid option! Try again\n");
		else { 
			*cacheBlockSize = stoi(aux);
			break;
			
		}
	}
	return;
}

void getDataType(int * dataType){
	while(1) {
		printf("\nChoose data type of elements: int (0), float (1), double (2)\n");
		cin >> *dataType;
		if (*dataType < 0 || *dataType > 2)
			printf("Invalid option! Try again\n");
		else
			break;
	}
	return;
}

void getOptimizationLevel(int * optimizationLevel){
	while(1) {
		printf("\nChoose compiler optimization level: -O0 (0), -O1 (1), -O2 (2), -O3 (3), -Ofast (4)\n");
		cin >> *optimizationLevel;
		if (*optimizationLevel < 0 || *optimizationLevel> 4)
			printf("Invalid option! Try again\n");
		else
			break;
	}
	return;
}

void compileSource(int* skip, int level) {
	cout << endl << "Compiling remaining programs... " << endl;
	
	int i;
	
	for (i = 0; i < g_programCount; ++i) {
		if (skip[i]) 
			continue;
		string str;
		str = "g++ -O";
		if (level <= 3) {
			str += to_string(level);
		} else {
			str = "g++ -Ofast -march=native -mfpmath=sse -frename-registers";
		}
		if (i == 0) {
			str += " -std=c++11 -o standardMult standard/standardMult.cpp -lpthread";
		} else if (i == 1) {
			str += " -std=c++11 -o optimizedMult standard/optimizedMult.cpp -lpthread";	
		} else if (i == 2) {
			str += " -std=c++11 -o optimizedMult2 standard/optimizedMult2.cpp -lpthread -mavx";
		} else if (i == 3) {
			str += " -std=c++11 -o optimizedMultParallel standard/optimizedMultParallel.cpp -lpthread -fopenmp";	
		} else if (i == 4) {
			str += " -std=c++11 -o optimizedMultParallel2 standard/optimizedMultParallel2.cpp -lpthread -mavx -fopenmp";
		}
		
		// Int
		if (g_dataType == 0) {
			if (i == 5) {
				str += " -std=c++11 -o strassenMult strassen/strassenMult_int.cpp -lpthread -DFORCE_SINGLE_CORE";
			} else if (i == 7) {
				str += " -std=c++11 -o strassenMultParallel strassen/strassenMult_int.cpp -lpthread";
			} else if (i == 9) {
				str += " -std=c++11 -o recursiveMult recursive/recursiveMult_int.cpp -lpthread -DFORCE_SINGLE_CORE";
			} else if (i == 11) {
				str += " -std=c++11 -o recursiveMultParallel recursive/recursiveMult_int.cpp -lpthread";
			} else if (i == 15) {
				str += " -std=c++11 -o standardOpenCL -I'/opt/AMDAPP/include' -I'/opt/AMDAPP/include/SDKUtil' -L'/opt/AMDAPP/lib/x86_64' openCL/standardOpenCL_int.cpp -lOpenCL";
			} else if (i == 16) {
				str += " -std=c++11 -o optimizedOpenCL -I'/opt/AMDAPP/include' -I'/opt/AMDAPP/include/SDKUtil' -L'/opt/AMDAPP/lib/x86_64' openCL/optimizedOpenCL_int.cpp -lOpenCL";
			} else if (i == 17) {
				str += " -std=c++11 -o AMDopenCL -I'/opt/AMDAPP/include' -I'/opt/AMDAPP/include/SDKUtil' -L'/opt/AMDAPP/lib/x86_64' openCL/AMDopenCL_int.cpp -lOpenCL";
			}
			
		// Float
		} else if (g_dataType == 1) {
			if (i == 5) {
				str += " -std=c++11 -o strassenMult strassen/strassenMult_float.cpp -lpthread -DFORCE_SINGLE_CORE";
			} else if (i == 6) {
				str += " -std=c++11 -o strassenMult2 strassen/strassenMult2_float.cpp -lpthread -DFORCE_SINGLE_CORE -mavx";
			} else if (i == 7) {
				str += " -std=c++11 -o strassenMultParallel strassen/strassenMult_float.cpp -lpthread";
			} else if (i == 8) {
				str += " -std=c++11 -o strassenMultParallel2 strassen/strassenMult2_float.cpp -lpthread -mavx";
			} else if (i == 9) {
				str += " -std=c++11 -o recursiveMult recursive/recursiveMult_float.cpp -lpthread -DFORCE_SINGLE_CORE";
			} else if (i == 10) {
				str += " -std=c++11 -o recursiveMult2 recursive/recursiveMult2_float.cpp -lpthread -DFORCE_SINGLE_CORE -mavx";
			} else if (i == 11) {
				str += " -std=c++11 -o recursiveMultParallel recursive/recursiveMult_float.cpp -lpthread";
			} else if (i == 12) {
				str += " -std=c++11 -o recursiveMultParallel2 recursive/recursiveMult2_float.cpp -lpthread -mavx";
			} else if (i == 15) {
				str += " -std=c++11 -o standardOpenCL -I'/opt/AMDAPP/include' -I'/opt/AMDAPP/include/SDKUtil' -L'/opt/AMDAPP/lib/x86_64' openCL/standardOpenCL_float.cpp -lOpenCL";
			} else if (i == 16) {
				str += " -std=c++11 -o optimizedOpenCL -I'/opt/AMDAPP/include' -I'/opt/AMDAPP/include/SDKUtil' -L'/opt/AMDAPP/lib/x86_64' openCL/optimizedOpenCL_float.cpp -lOpenCL";
			} else if (i == 17) {
				str += " -std=c++11 -o AMDopenCL -I'/opt/AMDAPP/include' -I'/opt/AMDAPP/include/SDKUtil' -L'/opt/AMDAPP/lib/x86_64' openCL/AMDopenCL_float.cpp -lOpenCL";
			} else if (i == 18) {
				str += " -std=c++11 -o clMath -I'/opt/clAmdBlas-1.10.321/include'  -I'/opt/AMDAPP/include' -I'/opt/AMDAPP/include/SDKUtil' -L'/opt/AMDAPP/lib/x86_64' -L'/opt/clAmdBlas-1.10.321/lib64' clMathFolder/clMath_float.cpp -lOpenCL -lclAmdBlas";
			}
			
		// Double
		} else if (g_dataType == 2) {
			if (i == 5) {
				str += " -std=c++11 -o strassenMult strassen/strassenMult_double.cpp -lpthread -DFORCE_SINGLE_CORE";
			} else if (i == 6) {
				str += " -std=c++11 -o strassenMult2 strassen/strassenMult2_double.cpp -lpthread -DFORCE_SINGLE_CORE -mavx";
			} else if (i == 7) {
				str += " -std=c++11 -o strassenMultParallel strassen/strassenMult_double.cpp -lpthread";
			} else if (i == 8) {
				str += " -std=c++11 -o strassenMultParallel2 strassen/strassenMult2_double.cpp -lpthread -mavx";
			} else if (i == 9) {
				str += " -std=c++11 -o recursiveMult recursive/recursiveMult_double.cpp -lpthread -DFORCE_SINGLE_CORE";
			} else if (i == 10) {
				str += " -std=c++11 -o recursiveMult2 recursive/recursiveMult2_double.cpp -lpthread -DFORCE_SINGLE_CORE -mavx";
			} else if (i == 11) {
				str += " -std=c++11 -o recursiveMultParallel recursive/recursiveMult_double.cpp -lpthread";
			} else if (i == 12) {
				str += " -std=c++11 -o recursiveMultParallel2 recursive/recursiveMult2_double.cpp -lpthread -mavx";
			} else if (i == 13) {
				str += " -std=c++11 -o openBlas -I'/home/edufgf/Área de Trabalho/OpenBlas/xianyi-OpenBLAS-347dded' openBlasFolder/openBlas.cpp -L'/home/edufgf/Área de Trabalho/OpenBlas/xianyi-OpenBLAS-347dded' -lopenblas -DFORCE_SINGLE_CORE";
			} else if (i == 14) {
				str += " -std=c++11 -o openBlasParallel -I'/home/edufgf/Área de Trabalho/OpenBlas/xianyi-OpenBLAS-347dded' openBlasFolder/openBlas.cpp -L'/home/edufgf/Área de Trabalho/OpenBlas/xianyi-OpenBLAS-347dded' -lopenblas";
			} else if (i == 15) {
				str += " -std=c++11 -o standardOpenCL -I'/opt/AMDAPP/include' -I'/opt/AMDAPP/include/SDKUtil' -L'/opt/AMDAPP/lib/x86_64' openCL/standardOpenCL_double.cpp -lOpenCL";
			} else if (i == 16) {
				str += " -std=c++11 -o optimizedOpenCL -I'/opt/AMDAPP/include' -I'/opt/AMDAPP/include/SDKUtil' -L'/opt/AMDAPP/lib/x86_64' openCL/optimizedOpenCL_double.cpp -lOpenCL";
			} else if (i == 17) {
				str += " -std=c++11 -o AMDopenCL -I'/opt/AMDAPP/include' -I'/opt/AMDAPP/include/SDKUtil' -L'/opt/AMDAPP/lib/x86_64' openCL/AMDopenCL_double.cpp -lOpenCL";	
			} else if (i == 18) {
				str += " -std=c++11 -o clMath -I'/opt/clAmdBlas-1.10.321/include'  -I'/opt/AMDAPP/include' -I'/opt/AMDAPP/include/SDKUtil' -L'/opt/AMDAPP/lib/x86_64' -L'/opt/clAmdBlas-1.10.321/lib64' clMathFolder/clMath_double.cpp -lOpenCL -lclAmdBlas";
			}
			
		}
		puts(str.c_str());
		system(str.c_str());
	}
	
	cout << endl << "Compilation completed!" << endl << endl;

}

void getRepeatCount(int * repeatCount) {
	printf("\nEnter the number of times each program will run. Higher number means more precise timings.\n");
	cin >> *repeatCount;
	return;
} 

// Generates random matrix A[m,k] and matrix B[k,n]
template <class E>
int generateRandomData(int m, int k, int n, int randomRangePos, int onlyPositives) {
	ofstream file_matrix_A; //File to output matrix A
	ofstream file_matrix_B; //File to output matrix B
	
	//int randomRangePos;	// Range for random positive numbers
	int randomRangeNeg; // Range for random negative numbers
	//int onlyPositives;	// If want only positive numbers
	E randNumber;       // randNumber of type E
	
	cout << endl;
	cout << "Generating random matrices..." << endl;
	
	if (onlyPositives == 1) {
		randomRangeNeg = 0;
	} else {
		onlyPositives = 0;
		randomRangeNeg = randomRangePos;   
		randomRangePos = randomRangePos*2 + 1; // This way we can generate the range properly
	}
	
	srand (time(NULL)); // Seed random function	
	
	file_matrix_A.open("matrix_A.txt"); // Create/Open file
	file_matrix_B.open("matrix_B.txt");
	
	if ( file_matrix_A.is_open() && file_matrix_B.is_open() ) {  // If could open/create both files
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < k; ++j) {
			    randNumber = ((rand() % (randomRangePos)) + onlyPositives) - randomRangeNeg;
			    if (randNumber!=0) 
			    	randNumber = randNumber + (1/(E)randNumber);	
		    	file_matrix_A << randNumber << " ";
		  	}
		  	file_matrix_A << endl;
		}
		
		for (int i = 0; i < k; ++i) {
			for (int j = 0; j < n; ++j) {
		    	randNumber = ((rand() % (randomRangePos))  + onlyPositives) - randomRangeNeg;
		    	if (randNumber!=0) 
			    	randNumber = randNumber + (1/(E)randNumber);	
		    	file_matrix_B << randNumber << " ";
		  	}
		  	file_matrix_B << endl;
		}
		
		file_matrix_A.close(); // Close file
		file_matrix_B.close(); 
		return 1;
    } else {
    	cout << "Couldn't open files to create random matrices!" << endl;
    	return 0;
    }
}

// Routine checks stored correct answer with answer generated by other algorithm.
template <class E>
int checkAnswer(int dim1, int dim2, Array2D<E> &matrix_C, string filename) {
	ifstream file_matrix_ans;
	filename = filename+"_matrix_C.txt";
	file_matrix_ans.open (filename, std::ifstream::in); //File to read matrix C
	double ans;
	double val;
	int ret = 1; // While return is 1, everything is fine.
	
	if (file_matrix_ans.is_open()) {
		for (int i = 0; i < dim1; ++i) {
			for (int j = 0; j < dim2; ++j) {
				file_matrix_ans >> val;
				
				// Converts stored answer to E.
				ans = (E) matrix_C(i,j);
				
				// If numbers are different, increment ret.
				if ( !(val + (E)g_EPS >= matrix_C(i,j) && matrix_C(i,j) + (E)g_EPS >= val) ){
					ret++;
				}
			}
		}
	} else ret = 0;

	file_matrix_ans.close();
	
	// Case ret is equal to 0.
	if (!ret) {
		cout << "Couldn't open matrix C file!" << endl;
		cout << "Checking routine won't work!" << endl;
	} else if (ret != 1) {
		cout << ret-1 << " distinct elements!" << endl;	// Prints the number of different answers.
		ret = 0;
	}
	
	return ret;
}

// Creates input file with matrices dimensions used by other programs.
int writeInputFile(int m, int k, int n, int cacheBlockSize, int dataType) {
	ofstream file_out;
	file_out.open("matrix_dim.txt");
	if (file_out.is_open()) {
		file_out << m << " " << k << " " << n << " " << cacheBlockSize << " " << dataType;
		file_out.close();
	} else return 0;
	return 1;
}

// Write timings results to a txt file.
void writeOutput(int m, int k, int n, int cacheBlockSize, int repeatCount, int* skip, double* timing, double* read_time, 
				 double* write_time, double* process_time, double* process_time_real, int* errProgram) {
	ofstream file_out;
	// Out file = bench_dim1_dim2_dim3.txt
	string filename = "bench_"+ to_string(m) + "_" + to_string(k) + "_" + to_string(n) + "_" + to_string(cacheBlockSize) + 
					   "_" + g_dataTypeStr + "_" + g_optimizationLevelStr + ".txt";
	file_out.open(filename);	
	
	if (file_out.is_open()) {
		
		file_out << endl << "===== Matrix-Matrix Multiplication =====" << endl << endl;
		file_out << "# Matrix Dimensions: " << m << " " << k << " " << n << endl;
		file_out << "# Cache Block Size: " << cacheBlockSize << endl;
		file_out << "# Repeat Count: " << repeatCount << endl;
		file_out << "# Data Type: " << g_dataTypeStr << endl;
		file_out << "# Optimization Level: " << g_optimizationLevelStr.c_str() << endl;
		file_out << endl << "========================================" << endl << endl;
	
		file_out << endl << "============ Read Timings ============" << endl << endl;
		file_out << std::fixed;	// Avoid scientific notation.
		file_out.precision(g_outputPrecision); // Timing numbers precision.

		for (int i = 0; i < g_programCount; ++i) {
			if (skip[i])
				continue;
			// Formatting and timing in seconds.
			file_out << std::setw(g_outputWidth) << std::left << program_name[i] << ": " << std::setw(7) << std::left
											 << read_time[i]/1000 << " seconds" << endl;
		}
		file_out << endl << "=======================================" << endl << endl;

		file_out << endl << "============ Write Timings ============" << endl << endl;
		file_out << std::fixed;	// Avoid scientific notation.
		file_out.precision(g_outputPrecision); // Timing numbers precision.

		for (int i = 0; i < g_programCount; ++i) {
			if (skip[i])
				continue;
			// Formatting and timing in seconds.
			file_out << std::setw(g_outputWidth) << std::left << program_name[i] << ": " << std::setw(7) << std::left
											 << write_time[i]/1000 << " seconds" << endl;
		}
		file_out << endl << "=======================================" << endl << endl;
	
		file_out << endl << "============ Process Timings ===========" << endl << endl;
		file_out << std::fixed;	// Avoid scientific notation.
		file_out.precision(g_outputPrecision); // Timing numbers precision.

		for (int i = 0; i < g_programCount; ++i) {
			if (skip[i])
				continue;
			// Formatting and timing in seconds.
			if (i >= 15) 
				file_out << std::setw(g_outputWidth) << std::left << program_name[i] << ": " << std::setw(7) << std::left
											 << process_time[i]/1000 << " (" << process_time_real[i]/1000 << ")" << " seconds" << endl;
			else
				file_out << std::setw(g_outputWidth) << std::left << program_name[i] << ": " << std::setw(7) << std::left
											 << process_time[i]/1000 << " seconds" << endl;
		}
		file_out << endl << "=======================================" << endl << endl;
		file_out << endl << "============ Final Results ============" << endl << endl;
		file_out << std::fixed;	// Avoid scientific notation.
		file_out.precision(g_outputPrecision); // Timing floating numbers resolution.
		for (int i = 0; i < g_programCount; ++i) {
			// If skipped algorithm, skip this iteration.
			if (skip[i])
				continue;
			file_out << std::setw(g_outputWidth) << std::left << program_name[i] << ": " << std::setw(7) << std::left
											 << timing[i]/1000 << " seconds" << endl;	
		}
		file_out << endl << "=======================================" << endl << endl;
		
		for (int i = 0; i < g_programCount; ++i) {
			if (skip[i]) continue;
			if (errProgram[i]) {
				file_out << "Error found! " << program_name[i] << " !" << endl;
			}
		}
	}

}

// Stores the answer matrix C. Executed only the first time, getting matrix generated by first algorithm executed.
template <class E>
int getFinalMatrix(int dim1, int dim2, Array2D<E> &matrix_C, string filename) {
	ifstream file_matrix_C;
	filename = filename+"_matrix_C.txt";
	file_matrix_C.open (filename, std::ifstream::in); //File to read matrix C
	
	if (file_matrix_C.is_open()) {
		for (int i = 0; i < dim1; ++i) {
			for (int j = 0; j < dim2; ++j) {
				file_matrix_C >> matrix_C(i, j); 
			}
		}
	} else return 0;
	file_matrix_C.close();
	return 1;
}

// Stores timing for one algorithm.
void getTiming(double* timing, struct timeval* time_start, struct timeval* time_end) {
	*timing = 0;
	
	*timing = (double)(time_end->tv_sec- time_start->tv_sec)*1000;
	*timing += (double)(time_end->tv_usec - time_start->tv_usec)/1000;
	
	return;				   
}

void readTiming(string filename, double * timing){
	ifstream time_file;	
	double aux;
	filename = filename+"_read_time.txt";
	time_file.open (filename, std::ifstream::in); //File to read timing
	time_file >> aux;
	*timing = *timing == -1 ? aux : min(*timing,aux); //Get minimum timing.
	time_file.close();
}

void writeTiming(string filename, double * timing){
	ifstream time_file;	
	double aux;
	filename = filename+"_write_time.txt";
	time_file.open (filename, std::ifstream::in); //File to read timing
	time_file >> aux;
	*timing = *timing == -1 ? aux : min(*timing,aux); //Get minimum timing.
	time_file.close();
}

void processTiming(string filename, double * timing){
	ifstream time_file;	
	double aux;
	filename = filename+"_process_time.txt";
	time_file.open (filename, std::ifstream::in); //File to read timing
	time_file >> aux;
	*timing = *timing == -1 ? aux : min(*timing,aux); //Get minimum timing.
	time_file.close();
}

void processTimingReal(string filename, double * timing){
	ifstream time_file;	
	double aux;
	filename = filename+"_process_time_real.txt";
	time_file.open (filename, std::ifstream::in); //File to read timing
	time_file >> aux;
	*timing = *timing == -1 ? aux : min(*timing,aux); //Get minimum timing.
	time_file.close();
}

int isPowerOfTwo(int n) {
	return !(n == 0) && !(n & (n - 1));
}

int main(int argc, char *argv[]) {	
	int m, k, n;
	int cacheBlockSize;
	int dataType;
	int optimizationLevel;
	int repeatCount;
	int readConsole;
	int randomRangePos;
	int onlyPositives;
	int firstRun = 1; // Flag to retrieve answer matrix C generated by the first algorithm executed.
	int ret;	

    struct timeval time_start, time_end; // Structs used by gettimeofday.
    double timing[g_programCount];		 // Stores timings for each algorithm.
    double read_time[g_programCount];	 // Stores timings for each algorithm.
    double write_time[g_programCount];	 // Stores timings for each algorithm.
    double process_time[g_programCount]; // Stores timings for each algorithm.
    double process_time_real[g_programCount]; // Stores timings for each algorithm.
    int skip[g_programCount];			 // Flag for algorithm skipping.
    int errProgram[g_programCount];		 // If matrix C is wrong.
    
    // Not skipping anyone at first.
    for (int i = 0; i < g_programCount; ++i) {
    	skip[i] = 0;
    	errProgram[i] = 0;
    	
    	// Reset timings.
    	read_time[i] = -1;
    	write_time[i] = -1;
    	process_time[i] = -1;	
    	process_time_real[i] = -1;
    }
    
    // execve arguments.
	char** new_argv;
    char *new_envp[] = {"LD_LIBRARY_PATH=/usr/lib/openblas-base/", (char*) 0};
    char *new_envp2[] = {"LD_LIBRARY_PATH=/opt/clAmdBlas-1.10.321/lib64", (char*) 0};
    new_argv = NULL;
    
    // Handle command line parameters.
	ret = handleParameters(argc, argv);
	
	// Invalid command line parameters.
	if (ret == 0) return 0;
	
	m = atoi(argv[1]);
	k = atoi(argv[2]);
	n = atoi(argv[3]);
	
	
	getConsole(&readConsole);
	
	if (readConsole) {
		getDataType(&dataType);
		g_dataType = dataType;
	
		getCacheSize(&cacheBlockSize);
	
		getOptimizationLevel(&optimizationLevel);
		g_optimizationLevelStr = "-O";
		g_optimizationLevelStr += to_string(optimizationLevel);
	
		getRepeatCount(&repeatCount);
		
		getRange(&randomRangePos, &onlyPositives);
		
	} else {
		g_dataType = atoi(argv[4]);	
		cacheBlockSize = atoi(argv[5]);
		
		optimizationLevel = atoi(argv[6]);
		g_optimizationLevelStr = "-O";
		g_optimizationLevelStr += to_string(optimizationLevel);
		
		repeatCount = atoi(argv[7]);
		
		randomRangePos = atoi(argv[8]);
		onlyPositives = atoi(argv[9]);
	}
	
	// Skipping strassen algorithm.
	if ( (!isPowerOfTwo(m) || !isPowerOfTwo(k) || !isPowerOfTwo(n)) || !(n==m && n==k)){
		skip[5] = 1; skip[6] = 1;
		skip[7] = 1; skip[8] = 1;
		// AMDOpenCL
		skip[17] = 1;
	}

	// Not available AVX for int on my machine (Ivy Bridge)
	if (g_dataType == 0) {
		skip[2] = 1; skip[4] = 1;
		skip[6] = 1; skip[8] = 1;
		skip[10] = 1;
		skip[12] = 1;
		skip[13] = 1;
		skip[14] = 1;
		// clMath
		skip[18] = 1;
	}

	// openBlas only double available.
	if (g_dataType == 1) { 
		skip[13] = 1;
		skip[14] = 1;
	}
	
	// Skip algo
	if (readConsole) {
		requestSkip(skip);
	} else {
		int argcnt = 10;
		char *op_c;
		for (int i = 0; i < g_programCount; ++i) {
			// Only asks for algorithms not skipped
			if (!skip[i]) {
				op_c = argv[argcnt++];
				if (*op_c=='y')
					skip[i] = 1;
			}	
		}
	}
	
	if (!writeInputFile(m, k, n, cacheBlockSize, g_dataType)) {
		cout << "Couldn't write matrix_dim.txt input file!" << endl;
		return 0;	
	}
	
	void* matrix_C;
    
    if (g_dataType == 0) {
    	matrix_C = (Array2D<int> *) matrix_C;
    	matrix_C = new Array2D<int>(m,n);
    	generateRandomData<int>(m, k, n, randomRangePos, onlyPositives);
    	g_dataTypeStr = "int";
    } else if (g_dataType == 1) {
    	matrix_C = (Array2D<float> *) matrix_C;
    	matrix_C = new Array2D<float>(m,n);
    	generateRandomData<float>(m, k, n, randomRangePos, onlyPositives);
    	g_dataTypeStr = "float";
    } else {
    	matrix_C = (Array2D<double> *) matrix_C;
    	matrix_C = new Array2D<double>(m,n);
    	generateRandomData<double>(m, k, n, randomRangePos, onlyPositives);
    	g_dataTypeStr = "double";
    }
    
	cout << "Matrices generated on files: matrix_A.txt and matrix_B.txt !" << endl;
	
	compileSource(skip, optimizationLevel);
	
	pid_t pid;
		
	// Stepping through all 6 algorithms.
	for (int currentProgram = 0; currentProgram < g_programCount; ++currentProgram) {
		// Skipping if set to skip.
		if (skip[currentProgram])
			continue;
		
		for (int repeat = 0; repeat < repeatCount; ++repeat) {
			cout << program_name[currentProgram] << " Starting #" << repeat+1 << "..." << endl;
			fflush(stdout);
		
			// Timing start.
			gettimeofday(&time_start, NULL);

			if (pid = fork() == 0) {
				// Child process executes other process.
				if (currentProgram == 18) {
					execve(program_name[currentProgram], new_argv, new_envp2);
				} else {
					execve(program_name[currentProgram], new_argv, new_envp);
				}
			} else {
				// Father process waits for it child to finish executing.
				wait(NULL);
			}
		
			// Timing end.
			gettimeofday(&time_end, NULL);
	
			cout << program_name[currentProgram] << " Done #" << repeat+1 << "!" << endl;
		
			// Calculating timing.
			getTiming(&timing[currentProgram], &time_start, &time_end);
		
			readTiming(program_name[currentProgram], &read_time[currentProgram]);
			writeTiming(program_name[currentProgram], &write_time[currentProgram]);
			processTiming(program_name[currentProgram], &process_time[currentProgram]);
			
			if (currentProgram >= 15)
				processTimingReal(program_name[currentProgram], &process_time_real[currentProgram]);	
			
			timing[currentProgram] = read_time[currentProgram] + write_time[currentProgram] + process_time[currentProgram];

			// First algorithm, retrieves the answer matrix C.
			if (firstRun) {
				if (g_dataType == 0) {
					if (!getFinalMatrix<int>(m, n, (* (Array2D<int>*) matrix_C), program_name[currentProgram])) {
						cout << "Couldn't open matrix C file!" << endl;
						cout << "Checking routine won't work!" << endl;		
					}
				} else if (g_dataType == 1) {
					if (!getFinalMatrix<float>(m, n, (* (Array2D<float>*) matrix_C), program_name[currentProgram])) {
						cout << "Couldn't open matrix C file!" << endl;
						cout << "Checking routine won't work!" << endl;		
					}
				} else {
					if (!getFinalMatrix<double>(m, n, (* (Array2D<double>*) matrix_C), program_name[currentProgram])) {
						cout << "Couldn't open matrix C file!" << endl;
						cout << "Checking routine won't work!" << endl;		
					}
				}
				firstRun = 0;
			} else if (repeat == 0) {
				// Check answers by comparing matrix C elements with new generated matrix.
				int ok;
				if (g_dataType == 0)
					ok = checkAnswer<int>(m, n, (* (Array2D<int>*) matrix_C), program_name[currentProgram]);
				else if (g_dataType == 1)
					ok = checkAnswer<float>(m, n, (* (Array2D<float>*) matrix_C), program_name[currentProgram]);	
				else
					ok = checkAnswer<double>(m, n, (* (Array2D<double>*) matrix_C), program_name[currentProgram]);
					
				if (!ok) {
					errProgram[currentProgram] = 1;
				}
			}
		}
	
	}

	cout << endl << "===== Matrix-Matrix Multiplication =====" << endl << endl;
	cout << "# Matrix Dimensions: " << m << " " << k << " " << n << endl;
	cout << "# Cache Block Size: " << cacheBlockSize << endl;
	cout << "# Repeat Count: " << repeatCount << endl;
	cout << "# Data Type: " << g_dataTypeStr << endl;
	cout << "# Optimization Level: " << g_optimizationLevelStr.c_str() << endl;
	cout << endl << "========================================" << endl << endl;
	
	cout << endl << "============= Read Timings ============" << endl << endl; 
	cout << std::fixed;	// Avoid scientific notation.
	std::cout.precision(g_outputPrecision); // Timing numbers precision.

	for (int i = 0; i < g_programCount; ++i) {
		if (skip[i])
			continue;
		// Formatting and timing in seconds.
		cout << std::setw(g_outputWidth) << std::left << program_name[i] << ": " << std::setw(7) << std::left
										 << read_time[i]/1000 << " seconds" << endl;
	}
	cout << endl << "=======================================" << endl << endl;

	cout << endl << "============ Write Timings ============" << endl << endl;
	cout << std::fixed;	// Avoid scientific notation.
	std::cout.precision(g_outputPrecision); // Timing numbers precision.

	for (int i = 0; i < g_programCount; ++i) {
		if (skip[i])
			continue;
		// Formatting and timing in seconds.
		cout << std::setw(g_outputWidth) << std::left << program_name[i] << ": " << std::setw(7) << std::left
										 << write_time[i]/1000 << " seconds" << endl;
	}
	cout << endl << "=======================================" << endl << endl;
	
	cout << endl << "============ Process Timings ===========" << endl << endl;
	cout << std::fixed;	// Avoid scientific notation.
	std::cout.precision(g_outputPrecision); // Timing numbers precision.

	for (int i = 0; i < g_programCount; ++i) {
		if (skip[i])
			continue;
		// Formatting and timing in seconds.
		if (i >= 15) 
			cout << std::setw(g_outputWidth) << std::left << program_name[i] << ": " << std::setw(7) << std::left
										 << process_time[i]/1000 << " (" << process_time_real[i]/1000 << ")" << " seconds" << endl;
		else
			cout << std::setw(g_outputWidth) << std::left << program_name[i] << ": " << std::setw(7) << std::left
										 << process_time[i]/1000 << " seconds" << endl;
	}
	cout << endl << "=======================================" << endl << endl;

	cout << endl << "============ Final Results ============" << endl << endl;
	cout << std::fixed;	// Avoid scientific notation.
	std::cout.precision(g_outputPrecision); // Timing numbers precision.

	for (int i = 0; i < g_programCount; ++i) {
		if (skip[i])
			continue;
		// Formatting and timing in seconds.
		cout << std::setw(g_outputWidth) << std::left << program_name[i] << ": " << std::setw(7) << std::left
										 << timing[i]/1000 << " seconds" << endl;
	}
	cout << endl << "=======================================" << endl << endl;
	
	for (int i = 0; i < g_programCount; ++i) {
		if (skip[i]) continue;
		if (errProgram[i]) {
			cout << "Error found! " << program_name[i] << " !" << endl;
		}
	}
	
	// Generates output file with timing results.
	writeOutput(m, k, n, cacheBlockSize, repeatCount, skip, timing, read_time, write_time, process_time, process_time_real, errProgram);
	
    return 0;
}
