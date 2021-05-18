simulator: parameters.o SimCycle.o SimSpGEMM.o SimOuterSPACE.o
	g++ parameters.o SimCycle.o SimSpGEMM.o SimOuterSPACE.o libramulator.a -o simulator -g -static

parameters.o: parameters.cpp
	g++ -c parameters.cpp -o parameters.o -O3 -g -std=c++17 $(FLAGS)

SimCycle.o: SimCycle.cpp common.h
	g++ -c SimCycle.cpp -o SimCycle.o -Iramulator/ -O3 -mtune=native -g -std=c++17 -DRAMULATOR

SimSpGEMM.o: SimSpGEMM.cpp common.h
	g++ -c SimSpGEMM.cpp -o SimSpGEMM.o -O3 -mtune=native -g -std=c++17

SimOuterSPACE.o: SimOuterSPACE.cpp common.h
	g++ -c SimOuterSPACE.cpp -o SimOuterSPACE.o -O3 -mtune=native -g -std=c++17 -Iramulator/ -DRAMULATOR

clean:
	rm -f *.o simulator
