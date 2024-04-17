#!/bin/bash
# Runs simulations in batch


#Defaults
S="10 50 100 200"
I=0
NOI="101"
N=0
TRACE=
HEAP=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so # libtbbmalloc_proxy.so # libtcmalloc.so libtcmalloc.so
R=1
CYCLES=50000000
T=""
s=""
D="."
FIRST="1"
CPT="-j config.json"
BATCHCONF="-l 0"
VERBO=""

#Cmdline options
while [ -n "$1" ]; do

	case "$1" in

	-D) 
		D="$2"

		echo "Simulation launched in dirs : $2"
		shift 
		;;

	-I)
		I="$2"

		echo "Simulations batch start on CPU : $I"

		shift
		;;

	-C)
		CYCLES="$2"
		echo "Cycles to simulate : $CYCLES"
		shift
		;;

    -N)
		N="$2"

		echo "Number of CPU per sim : $N"

		shift
		;;

    -S)
		S="$2"

		echo "Sequences to learn (last): $S"

		shift
		;;
	-W)
		FIRST="$2"

		echo "Starting sequence : $FIRST"

		shift
		;;
    -R)
		R="$2"

		echo "Runs per simulation : $R"

		shift
		;;
   -O) echo "Run with TRACE disabled: unlimited periodic! "
		TRACE=""
		;;	
		
	-K) 
		CPT="-k cpt"
		echo "Primed with checkpoint ./cpt/"
		;;

   -T) echo "Tracing Enabled (-T)"
        T="-t"
        ;;
    -b)
		BATCHCONF="$2"
		echo "Batch config $BATCHCONF"
		shift
		;;
	-V)
		VERBO="-v"

		echo "Verbosity enabled (all CC dumps HEROUT)"
		shift
		;;
	*) echo "Option $1 not recognized" 
        echo "-S Sequences to learn $S"
        echo "-W starting sequence $FIRST"
        echo "-R Runs per simulation $R"
        echo "-N Number of 2w SMT CPU per sim $N (0 run seq)"
        echo "-I Simulations batch start on CPU $I"
        echo "-C cycles to simulate $CYCLES"
        echo "-D dirs to launch the simulation ($D)"
        echo "-O Non-TRACE run faster (TRACE=$TRACE)"
        echo "-T dump detailed trace (accoding config)"
		echo "-b batch learning config (e.g., \"-l 10 -z 1 -i 10\" batch lengh 10,\
		 sliding 1 each 10 iterations [if iter=0 determines internal cortex to move forward])"
		echo "-V verbosity on"
        exit
        ;;

	esac
    
	shift

done

CPUS=$(expr  `nproc` / 2)
for DIR in $D ;
do
	for IDX in $(seq 1 $R)
	do
		SEED=13937563$IDX
		echo $SEED
		for LAST in $S ; 
		do
			for db in $NOI ;
			do
				if [ $N != 0 ]
				then
						cd $DIR && export TRACE=$TRACE && LD_PRELOAD=$HEAP  taskset -c $I-$(expr $I + $N - 1),$(expr $I + $CPUS)-$(expr $I + $N - 1 + $CPUS)  python ./pyt/speech/realASR.py \
						-c $CYCLES $T -w $FIRST -W $LAST -o real$FIRST-$LAST-i$IDX $CPT $s -s $SEED $BATCHCONF $VERBO >& log_${PRE}_$LAST-i$IDX &
					I=$((($I+$N)%$CPUS))
				else
					cd $DIR && export TRACE=$TRACE && LD_PRELOAD=$HEAP  taskset -c $I python ./pyt/speech/realASR.py \
						-0 -c $CYCLES $T -w $FIRST -W $LAST -o real$FIRST-$LAST-i$IDX $CPT $s -s $SEED $BATCHCONF $VERBO >& log_${PRE}_$LAST-i$IDX &
					I=$(($I+1))
				fi
			done
		done 
	done
done
