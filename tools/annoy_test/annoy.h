#ifndef __ANNOY_H__
#define __ANNOY_H__

#define HANDLE void*

int annoy_init(HANDLE* handle, int dimension);

int annoy_add_item(HANDLE handle, int index, float* data);

int annoy_build_tree(HANDLE handle, int num_tree, bool save_tree);

int annoy_get_nns(HANDLE handle, int k, int n, float* query, int* res_indexs);

#endif  // __ANNOY_H__
