#!/bin/bash
set -eu

# run by ./histogram.py no_test_runs Some description of your current code
# $./histogram.py 50 Without diversity keeping, population 100

# your number here -------------------------
student_number=r0829194
# your number here -------------------------

if ! [ -f $student_number.py ]; then
    echo "Can't see python script. Check student number."
    exit
fi

function exitus() {
    echo "Exiting program."
    rm -v $script $py_csv
    cd $root
}

function cleanup() {
    echo "** Trapped CTRL-C"
    rm -v $script $py_csv
    cd $root
}
trap cleanup INT

if [ $# -lt 1 ]; then
    echo "Forgot to give number of test runs?"
    exit
fi
no_test_runs=$1
shift

if ! [ "$no_test_runs" -eq "$no_test_runs" ] 2>/dev/null; then
    echo "Need a number as first argument!"
    exit
fi

root=$(pwd)
temp=$root/temp
out=$root/out

if ! [ -d $temp ]; then
    mkdir $temp
    [ $? -ne 0 ] && exit
fi
if ! [ -d $out ]; then
    mkdir $out
    [ $? -ne 0 ] && exit
fi

pid=$$
my_name=histogram_${pid}
echo Histogram of $student_number.py, $pid - $no_test_runs runs: $@ >> $out/histogram_PID.log 

script=$temp/${student_number}_${pid}_histogram_copy.py 
py_csv_base=${student_number}_${pid}
py_csv=$temp/$py_csv_base.csv
hist_csv=$temp/$my_name.csv

cp $root/$student_number.py $script
tour_csv_name=$(grep -E '.*"tour[0-9]+.csv".*' $script | sed -E -e 's/.*"(.+)".*/\1/')
tour_csv=$temp/$tour_csv_name
cp $root/$tour_csv_name $tour_csv
cp $root/Reporter.py $temp/Reporter.py
cp $root/histogram.py $temp/histogram.py
sed -E -i -e "s/$student_number/$py_csv_base/g" $script
[ $? -ne 0 ] && exitus

echo "Run, Iterations, Time, Mean value, Best value" > $hist_csv
cd $temp
echo "Running $student_number.py $no_test_runs times."
for (( i=1; i<=$no_test_runs; i++ ))
do
    python3 $script > /dev/null
    [ $? -ne 0 ] && exitus
    tail -1  $py_csv | sed -E -e "s/(([0-9.]+,){3}[0-9.]+).*/$i,\1/" >> $hist_csv
    [ $? -ne 0 ] && exitus
    ((i%10==0)) && echo "Done $i runs."
done
echo "Done with runs."

python3 $temp/histogram.py $hist_csv
mv $temp/histogram.png $out/$my_name.png

exitus
