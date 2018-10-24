#include "annoy.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <stdexcept>

using namespace std;

bool read_point(FILE *file, float** vec) {
  int d;
  if (fread(&d, sizeof(int), 1, file) != 1) {
    return false;
  }
  assert(d == 512);
  float *buf = new float[d];
  if (fread(buf, sizeof(float), d, file) != (size_t)d) {
    throw runtime_error("can't read a point");
  }
  *vec = buf;
  return true;
}

int main() {
  HANDLE tree;
  annoy_init(&tree, (int)512);
  FILE* file = fopen("megaface.dat", "rb");
  if (!file) {
    cout << "file open failed.";
    return -1;
  }
  float* vec = NULL;
  int index = 0;
  while (read_point(file, &vec)) {
    annoy_add_item(tree, index, vec);
    ++index;
  }
  annoy_build_tree(tree, 80, true);
  int k = 10;
  int* res_indexs = new int[k];

  FILE* file1 = fopen("probe.dat", "rb");
  if (!file1) {
    cout << "open failed." << endl;
    return -1;
  }
  float* query = NULL;
  int nn = 0;
  while (read_point(file1, &query)) {
    annoy_get_nns(tree, k, k*5, query, res_indexs);
  }

  return 0;
}
