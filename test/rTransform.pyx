# distutils: language = c++
# distutils: sources = RTransform.cpp

# Cython interface file for wrapping the object
#

from libcpp.vector cimport vector

ctypedef unsigned char uint8_t

# c++ interface to cython
cdef extern from "RTransform.h":
  cdef cppclass RTransformer:
        RTransformer() except +
        int x0, y0, x1, y1
        vector[double] RTransform(vector[vector[uint8_t]], int, int, int)

# creating a cython wrapper class
cdef class PyRTransform:
    cdef RTransformer *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new RTransformer()
    def __dealloc__(self):
        del self.thisptr
    def rTransform(self, sv, cols, rows, N):
        return self.thisptr.RTransform(sv, cols, rows, N)