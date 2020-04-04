#When `source` command is used to execute this script, exits right away and starts a sub-shell to execute it.
if [[ $0 == "bash" ]]; then
    bash compare.sh
    return
fi

set -e
trap 'if [[ $? != 0 ]]; then echo "An error occured. Exiting..."; sleep 1; fi' EXIT

make read_mnist.out

time python3 check.py > __a
time ./read_mnist.out > __b

command vimdiff __a __b

rm __a __b

echo
echo "Done"

set +e

