set terminal qt size 640, 480 position 1000, 300
# set terminal qt size 640, 480 position 280, 300

# set terminal postscript eps color size 5, 3.5 colortext font "Helvetica, 14"
# set terminal gif size 640, 480 font "Helvetica, 12" animate delay 10
# set terminal png truecolor font "Helvetica, 12" size 640, 480

# set table ""
FILENAME = "fig_3_2.eps"
# set output FILENAME

# set title ""
# set xlabel ""
# set ylabel ""
# set format x "%.0f"
# set format y "%.0f"

# unset key
set key left
# set grid
# set size square

# set logscale x
# set logscale y

set xrange [-2:2]
set yrange [-0.05:1.5]

# set timestamp "%Y/%m/%d (%a) %H:%M:%S"

set pointsize 1.0
# set samples 200

# set multiplot layout 2, 2

step(x) = (x > 0)
sigmoid(x) = 1 / (1 + exp(-x))
ReLU(x) = (x > 0 ? x : 0)

plot step(x) title "step function", sigmoid(x) title "sigmoid function", ReLU(x) title "ReLU"












# unset multiplot
pause -1 "Press any key: "

