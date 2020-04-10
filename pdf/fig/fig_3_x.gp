set terminal qt size 1280, 480 position 1000, 300
# set terminal qt size 640, 480 position 280, 300

# set terminal postscript eps color size 5, 3.5 colortext font "Helvetica, 14"
# set terminal gif size 640, 480 font "Helvetica, 12" animate delay 10
# set terminal png truecolor font "Helvetica, 12" size 640, 480

# set table ""
FILENAME = ""
# set output FILENAME

# set title ""
# set xlabel ""
# set ylabel ""
# set format x "%.0f"
# set format y "%.0f"

unset key
# set grid
# set size square

# set logscale x
# set logscale y

set xrange [0.5:9.5]

# set timestamp "%Y/%m/%d (%a) %H:%M:%S"

set pointsize 1.0
# set samples 200

# set multiplot layout 2, 2
set multiplot layout 1, 2

$DATA << EOD
1.00 0.07805982211475883
1.05 0.08206203477749992
1.08 0.08456119580192598
1.10 0.08626944527138976
1.24 0.09923348253762715
1.30 0.10536973839942328
1.31 0.10642872187595255
1.38 0.1141456749323887
1.40 0.11645157052428143
1.49 0.1274183137647524
EOD

line_width = 1.2

set style fill solid 0.3 border 1

set yrange [0:1.5]
plot $DATA using 1 with boxes linewidth line_width

set yrange [0:0.13]
plot $DATA using 2 with boxes linewidth line_width

#Python 3
# A = [1.00, 1.05, 1.08, 1.10, 1.24, 1.30, 1.31, 1.38, 1.40, 1.49, ]
# expA = np.sum(np.exp(np.array(A)))
# for i in A:
#     print(exp(i) / expA)

# unset multiplot
pause -1 "Press any key: "

