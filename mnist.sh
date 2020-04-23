set -e

if (( $# <= 2 )); then
    echo "Usage: sh mnist.sh [<log dir>] <seed> [<epoch> [<num_node_of_hidden_layer>...] ]"
    exit 0
fi

output_dir="result"
if [[ -d "$1" ]]; then
    output_dir=${1%/}
    shift
fi

seed=$1
epoch=$2
shift 2
num_node_of_hidden_layer=( $@ )

output_file="${output_dir}/log_${seed}_${epoch}"
if [[ ! -z ${num_node_of_hidden_layer} ]]; then
    output_file="${output_file}_${num_node_of_hidden_layer[0]}"
    for i in $(seq 1 $(( ${#num_node_of_hidden_layer[@]} - 1))); do
        output_file="${output_file}-${num_node_of_hidden_layer[$i]}"
    done
fi
output_file="${output_file}.txt"
if [[ -f "${output_file}" ]]; then
    output_file="${output_file}_$(mktemp --dry-run XXXXXX)"
fi

binary_name="mnist.out"

make

date
echo "time ./${binary_name} ${seed} ${epoch} ${num_node_of_hidden_layer} > \"${output_file}\""

trap 'echo "Inturrupted." && rm -v "${output_file}"' SIGINT
time "./${binary_name}" ${seed} ${epoch} ${num_node_of_hidden_layer} > "${output_file}"

