#!/bin/bash

# -Int- 10 programs
#"standardMult", 
#"optimizedMult",  
#"optimizedMultParallel", 
#"strassenMult",  
#"strassenMultParallel", 
#"recursiveMult", 
#"recursiveMultParallel",  
#"standardOpenCL", 
#"optimizedOpenCL", 
#"AMDopenCL", 

# -Float- 17 programs
#"standardMult", 
#"optimizedMult", 
#"optimizedMult2", 
#"optimizedMultParallel", 
#"optimizedMultParallel2", 
#"strassenMult", 
#"strassenMult2", 
#"strassenMultParallel", 
#"strassenMultParallel2", 
#"recursiveMult", 
#"recursiveMult2", 
#"recursiveMultParallel", 
#"recursiveMultParallel2", 
#"standardOpenCL", 
#"optimizedOpenCL", 
#"AMDopenCL", 
#"clMath"};

# -Double- 19 programs
#"standardMult", 
#"optimizedMult", 
#"optimizedMult2", 
#"optimizedMultParallel", 
#"optimizedMultParallel2", 
#"strassenMult", 
#"strassenMult2", 
#"strassenMultParallel", 
#"strassenMultParallel2", 
#"recursiveMult", 
#"recursiveMult2", 
#"recursiveMultParallel", 
#"recursiveMultParallel2", 
#"openBlas", 
#"openBlasParallel", 
#"standardOpenCL", 
#"optimizedOpenCL", 
#"AMDopenCL", 
#"clMath"};

max=2048
repeat=5
range=10
positive=1

p1=n 
p2=n 
p3=n 
p4=n
p5=n
p6=n
p7=n
p8=y
p9=y
p10=y
p11=y
p12=y
p13=y
p14=y
p15=y
p16=y
p17=y
p18=y
p19=y

#Ordem das combinações
#1 - Tipo (int/float/double)
#2 - Tamanho(256, 512, 1024..., 8192)
#3 - Otimização(O0, O2, Ofast)
#4 - Block Size (32, 64, 128)
# No ultimo block size(128) ativar Mult GPU e openBlas
# Ultimo block int
# -- "standardOpenCL", "optimizedOpenCL", "AMDopenCL" 
# Ultimo block Float
# -- "standardOpenCL", "optimizedOpenCL", "AMDopenCL", "clMath" 
# Ultimo block Double
# -- "openBlas", "openBlasParallel", "standardOpenCL", "optimizedOpenCL", "AMDopenCL", "clMath" 

for (( tipo=1; tipo<=2; tipo++ ))
do
	for (( tam=256; tam<=$max; tam*=2 ))	
	do
		for (( otimizacao=0; otimizacao<=4; otimizacao+=2 ))
		do
			for (( blocksize=32; blocksize<=128; blocksize*=2 ))
			do
				p1=n;
				repeat=5;
				#Stop standardMult on big matrices
				if [ $tam -ge 4096 ]; then 
					p1=y;		
				fi
				
				#Repeat count modification
				if [ $tam -ge 2048 ]; then
					repeat=3;
				fi
				
				p8=y;
				p9=y;
				p10=y; 
				p11=y; 
				p12=y; 
				p13=y;
				p14=y; 
				p15=y; 
				p16=y; 
				p17=y;
				p18=y;
				p19=y;
				
				#Int
				if [ $tipo == 0 ]; then
					# Run GPU on last blocksize
					if [ $blocksize == 128 ]; then
						p8=n;
						p9=n;
						p10=n;
					fi
					./test-mult $tam $tam $tam $tipo $blocksize $otimizacao $repeat $range $positive $p1 $p2 $p3 $p4 $p5 $p6 $p7 $p8 $p9 $p10 < in.txt
				fi
				
				#Float
				if [ $tipo == 1 ]; then
					p8=n;
					p9=n;
					p10=n; 
					p11=n; 
					p12=n; 
					p13=n;
					# Run GPU on last blocksize
					if [ $blocksize == 128 ]; then
						p14=n; 
						p15=n; 
						p16=n; 
						p17=n;
					fi
					./test-mult $tam $tam $tam $tipo $blocksize $otimizacao $repeat $range $positive $p1 $p2 $p3 $p4 $p5 $p6 $p7 $p8 $p9 $p10 $p11 $p12 $p13 $p14 $p15 $p16 $p17 < in.txt
				fi
				
				#Double
				if [ $tipo == 2 ]; then
					p8=n;
					p9=n;
					p10=n; 
					p11=n; 
					p12=n; 
					p13=n;
					# Run GPU on last blocksize
					if [ $blocksize == 128 ]; then
						p14=n; 
						p15=n; 
						p16=n; 
						p17=n;
						p18=n;
						p19=n;
					fi
					./test-mult $tam $tam $tam $tipo $blocksize $otimizacao $repeat $range $positive $p1 $p2 $p3 $p4 $p5 $p6 $p7 $p8 $p9 $p10 $p11 $p12 $p13 $p14 $p15 $p16 $p17 $p18 $p19 < in.txt
				fi
			done
		done
	done
done

