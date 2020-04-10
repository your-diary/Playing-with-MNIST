set -e

if (( $# <= 2 )); then
    ./mnist.out
    exit 0
fi

seed=$1
epoch=$2
shift 2
num_node_of_hidden_layer=( $@ )

output_file="result/log_${seed}_${epoch}_${num_node_of_hidden_layer[0]}"
for i in $(seq 1 $(( ${#num_node_of_hidden_layer[@]} - 1))); do
    output_file="${output_file}-${num_node_of_hidden_layer[$i]}"
done
output_file="${output_file}.txt"

date
echo "time ./mnist.out ${seed} ${epoch} ${num_node_of_hidden_layer} > \"${output_file}\""

trap 'echo "Inturrupted." && rm -v "${output_file}"' SIGINT
time ./mnist.out ${seed} ${epoch} ${num_node_of_hidden_layer} > "${output_file}"

