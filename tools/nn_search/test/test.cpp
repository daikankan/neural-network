#include <iostream>
#include "../top_k.hpp"

using namespace std;

int main() {
  TopK topk(1000000, 512, "/home/dkk/projects/annoy-sdk/megaface.dat");
  float* query = new float[512];
  for (int i = 0; i < 512; ++i) {
    query[i] = 0.5;
  }
  int* indices = new int[50];
  topk.get_top_k(query, 50, indices);
  for (int i = 0; i < 50; ++i) {
    cout << indices[i] << " ";
  }
  cout << endl;
  delete[] query;
  delete[] indices;
  return 0;
}
