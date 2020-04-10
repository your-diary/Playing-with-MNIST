#Currently, recognition accuracy for digits drawn via this application is very low.
#This is probably because each pixel has not a real value ranged over [0,1] but the value `0` or `1`.

import tkinter as tk
from tkinter import ttk
import subprocess

root = tk.Tk()

root.title("draw_digit")
root.resizable(0, 0) #forbids resize

#initial coordinates of a window
initial_coordinate_x = 0
initial_coordinate_y = 0
root.geometry("+" + str(initial_coordinate_x) + "+" + str(initial_coordinate_y))

pad_y = 5 #vertical margin between objects (i.e. buttons, labels, etc.)

class CanvasMatrix():

    def __init__(self, num_row, num_column):
        self.num_row_ = num_row
        self.num_column_ = num_column
        self.canvas_matrix_ = self.__create_matrix_(self.num_row_, self.num_column_)

    def __create_matrix_(self, num_row, num_column):
        ret = []
        for i in range(num_row):
            ret.append([])
            for j in range(num_column):
                ret[i].append(0)
        return ret

    def set_dot_(self, x, y, value = 1):

        if (y >= self.num_row_ or x >= self.num_column_): #out of range
            return False

        # print("({:2d}, {:2d})".format(y, x))

        if (self.canvas_matrix_[y][x] == value):
            return False
        else:
            self.canvas_matrix_[y][x] = value
            return True

    def clear_(self):
        for i in range(self.num_row_):
            for j in range(self.num_column_):
                self.canvas_matrix_[i][j] = 0

    def print_canvas_matrix_(self):
        for i in range(self.num_row_):
            for j in range(self.num_column_):
                if (self.canvas_matrix_[i][j] == 1):
                    print("■", end = "")
                else:
                    print(" ", end = "")
                print(" ", end = "")
            print()

    def to_string_(self):
        ret = ""
        for i in range(self.num_row_):
            for j in range(self.num_column_):
                ret += str(self.canvas_matrix_[i][j]) + " "
        return ret

#canvas {

#apparent dimensions
canvas_height = 400
canvas_width = 400

#true dimensions
num_pixel_height = 28
num_pixel_width = 28

canvas_matrix = CanvasMatrix(num_pixel_height, num_pixel_width)

dot_height = canvas_height / num_pixel_height
dot_width = canvas_width / num_pixel_width

#See |http://www.science.smith.edu/dftwiki/images/3/3d/TkInterColorCharts.png| for the color list.
canvas_background_color = "gray70"

canvas = tk.Canvas(root, height = canvas_height, width = canvas_width, bg = canvas_background_color)
canvas.grid(row = 0, columnspan = 2);

canvas_grid_line_color = "gray40"
def draw_grid_lines():
    for i in range(num_pixel_width):
        canvas.create_line(i * dot_width, 0, i * dot_width, canvas_width, fill = canvas_grid_line_color)
    for i in range(num_pixel_height):
        canvas.create_line(0, i * dot_height, canvas_height, i * dot_height, fill = canvas_grid_line_color)

draw_grid_lines()

#} canvas

#bind {

def put_dot(event):
    dot_color = "black"
    if (canvas_matrix.set_dot_(int(event.x // dot_width), int(event.y // dot_height))): #If no dot has already been placed here.
        x = (event.x // dot_width + 0.5) * dot_width
        y = (event.y // dot_height + 0.5) * dot_height
        canvas.create_rectangle(x - dot_width / 2, y - dot_height / 2, x + dot_width / 2, y + dot_height / 2, fill = dot_color, outline = dot_color)

canvas.bind("<Button-1>", put_dot)
canvas.bind("<Button1-Motion>", put_dot)

def remove_dot(event):
    if (canvas_matrix.set_dot_(int(event.x // dot_width), int(event.y // dot_height), value = 0)): #If a dot exists here.
        x = (event.x // dot_width + 0.5) * dot_width
        y = (event.y // dot_height + 0.5) * dot_height
        canvas.create_rectangle(x - dot_width / 2, y - dot_height / 2, x + dot_width / 2, y + dot_height / 2, fill = canvas_background_color, outline = canvas_grid_line_color)

canvas.bind("<Button-3>", remove_dot)
canvas.bind("<Button3-Motion>", remove_dot)

def clear_canvas(event = 0):
    canvas.delete("all")
    draw_grid_lines()
    canvas_matrix.clear_()
    inferred_digit.set("")

canvas.bind("<Button-2>", clear_canvas)

clear_button = tk.Button(root, text = "Clear Canvas", command = clear_canvas)
clear_button.grid(row = 2, column = 1, pady = pad_y)

#} bind

#inference {

binary_name = "./infer_digit.out"

def infer_digit(event):
    pipe = subprocess.Popen([binary_name], stdout = subprocess.PIPE, stdin = subprocess.PIPE)
    stdout, stderr = pipe.communicate(input = "{}\n".format(canvas_matrix.to_string_()).encode("utf-8"))
    if (stderr):
        print(stderr)
    inferred_digit.set(stdout[:-1].decode("utf-8")) #`[:-1]` omits the redundant newline.

canvas.bind("<ButtonRelease-1>", infer_digit)
canvas.bind("<ButtonRelease-3>", infer_digit)

print_matrix_button = tk.Button(root, text = "Print Matrix", command = lambda: canvas_matrix.print_canvas_matrix_())
print_matrix_button.grid(row = 2, column = 0, pady = pad_y)

inferred_digit = tk.StringVar()
inferred_digit.set("")
result_label = tk.Label(root, textvariable = inferred_digit, width = 2, font = ("", 20), bg = "black", fg = "white")
result_label.grid(row = 4, column = 1, pady = pad_y, sticky = "w")
separator = ttk.Separator(root)
separator.grid(row = 3, columnspan = 2, pady = pad_y, sticky = "ew")
result_title_label = tk.Label(root, text = "Inferred Digit:   ")
result_title_label.grid(row = 4, column = 0, pady = pad_y, sticky = "e")

#} inference

welcome_message = """
-------- Instructions ---------
  Left Click: put a dot
 Right Click: remove a dot
Middle Click: clear the canvas
-------------------------------
"""
print(welcome_message)

root.mainloop()

