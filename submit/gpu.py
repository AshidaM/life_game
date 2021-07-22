#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy
import curses
from curses import wrapper
from pycuda.compiler import SourceModule

MAT_SIZE_X = 100
MAT_SIZE_Y = 100
BLOCKSIZE = 32

cell_value = lambda world, height, width, y, x: world[y % height, x % width]
row2str = lambda row: ''.join(['O' if c != 0 else '-' for c in row])

#GPUコード#
mod = SourceModule("""
__global__ void calc_next_cell_state_gpu(const int *world, int *next_world, const int height, const int width){
    
    
    int mat_x = threadIdx.x + blockIdx.x * blockDim.x;
    int mat_y = threadIdx.y + blockIdx.y * blockDim.y;

    if(mat_x >= width){
        return;
    }
    if(mat_y >= height){
        return;
    }
    
    int index = mat_y * width + mat_x;
    int num = 0;
    float cell = world[index];
    float next_cell = cell;
    num += world[((mat_y - 1) % height) * width + (mat_x - 1) % width];
    num += world[((mat_y - 1) % height) * width +  mat_x      % width];
    num += world[((mat_y - 1) % height) * width + (mat_x + 1) % width];
    num += world[  mat_y      % height  * width + (mat_x - 1) % width];
    num += world[  mat_y      % height  * width + (mat_x + 1) % width];
    num += world[((mat_y + 1) % height) * width + (mat_x - 1) % width];
    num += world[((mat_y + 1) % height) * width +  mat_x      % width];
    num += world[((mat_y + 1) % height) * width + (mat_x + 1) % width];   

    if(cell == 0 && num == 3){
         next_cell = 1;
    }
    else if(cell == 1 && (num == 2 || num == 3)){
        next_cell = 1;
    }
    else{
        next_cell = 0;
    }
    next_world[index] = next_cell;
}
""")
calc_next_cell_state_gpu = mod.get_function("calc_next_cell_state_gpu")

def print_world(stdscr, gen, world):
    '''
    盤面をターミナルに出力する
    '''
    stdscr.clear()
    stdscr.nodelay(True)
    scr_height, scr_width = stdscr.getmaxyx()
    height, width = world.shape
    height = min(height, scr_height)
    width = min(width, scr_width - 1)
    for y in range(height):
        row = world[y][:width]
        stdscr.addstr(y, 0, row2str(row))
    stdscr.refresh()

def calc_next_world_gpu(world, next_world):
    '''
    現行世代の盤面の状況を元に次世代の盤面を計算する
    '''
    block = (BLOCKSIZE, BLOCKSIZE, 1)
    grid = ((MAT_SIZE_X + block[0] - 1 ) // block[0], (MAT_SIZE_Y + block[1] - 1 ) // block[1])
    height, width = world.shape
    calc_next_cell_state_gpu(cuda.In(world), cuda.Out(next_world), numpy.int32(height), numpy.int32(width), block = block, grid = grid)

def gol(stdscr, height, width):
    # 状態を持つ2次元配列を生成し、0 or 1 の乱数で初期化する。
    world = numpy.random.randint(2, size=(height, width), dtype=numpy.int32)
    gen = 0
    while True:
        print_world(stdscr, gen, world)
        next_world = numpy.empty((height, width), dtype=numpy.int32)
        calc_next_world_gpu(world, next_world)
        world = next_world.copy()
        gen += 1

def main(stdscr):
    gol(stdscr, MAT_SIZE_X, MAT_SIZE_Y)
    
if __name__ == '__main__':
    curses.wrapper(main)

