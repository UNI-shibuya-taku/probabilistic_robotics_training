#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
using namespace std;
#include <list>
#include <math.h>

int partition(int *a, int p, int r){
  int x = a[r]; // 一番最後を記録
  int i = p-1; // 最初の番号
  int tmp = 0;
  int tmp2 = 0;
  for(int j = p; j < r; j++){
    if(a[j] <= x){
      i++;
      tmp = a[i];
      a[i] = a[j];
      a[j] = tmp;
      tmp = 0;
    }
  }
  tmp2 = a[i+1];
  a[i+1] = a[r];
  a[r] = tmp2;
  return i+1;
}

int main(){
  int input;
  int p = 0;
  cin >> input;
  int s = input;
  int a[input];
  for(int i = 0; i < input; i++){
    int t = 0;
    cin >> t;
    a[i] = t;
  }
  printf("a[-1]:%d\n",a[-1]);
  printf("%d\n",a[-2] );

  int mid = partition(a,p,input-1);

  printf("mid: %d\n", mid);

  for(int j = 0; j < s; j++){

    if(j == mid){
      printf("[%d] ", a[j]);
    }
    else{
      printf("%d ",a[j]);
    }
  }
  if(mid == input-1){
    printf("[%d]\n",a[input-1]);
  }
  else{
    printf("%d\n",a[input-1]);
  }
  printf("\n");
}
