#include<iostream>
#include<fstream>
#include<stdlib.h>
#include<ctime>

#include<sys/time.h>

#include<immintrin.h>  // AVX

#include<thread>  // C++11中引入的thread类
#include<mutex>
#include<condition_variable>

#include<omp.h>

#include<cmath>

using namespace std;

void reset(float** &a, int n);
void show(float** a, int n);
void GE(float** a, int n);
void C_GE(float** a, int n);

int n = 1000;
int lim = 1;
float** mat = nullptr;
const int THREAD_NUM = 7;  // 线程数量
int chunksize = 0;
ofstream out("output.txt");
ofstream sampleout("sampleout.txt");
//ifstream in("input.txt");

//--------------------------------------消去算法--------------------------------------

void GE(float** a, int n) {  // 标准的高斯消去算法, Gauss Elimination缩写
	for (int k = 0; k < n; k++)
	{
		for (int j = k + 1; j < n; j++)
		{
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; j++)
			{
				a[i][j] -= a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void C_GE(float** a, int n) {  // 高斯消去算法的Cache优化版本
	//__m128 va;
	float t1, t2;  // 使用两个浮点数暂存数据以减少程序中地址的访问次数
	for (int k = 0; k < n; k++)
	{
		t1 = a[k][k];
		for (int j = k + 1; j < n; j++)
		{
			a[k][j] = a[k][j] / t1;
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			t2 = a[i][k];
			for (int j = k + 1; j < n; j++)
			{
				a[i][j] -= t2 * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

//--------------------------------------消去算法(OMP)--------------------------------------

void C_GE_OMP_Dynamic(float** a, int n) {  // 高斯消去算法的Cache优化版本 _OMP
	//__m128 va;
	float t1, t2;  // 使用两个浮点数暂存数据以减少程序中地址的访问次数
	int i, j, k;
#pragma omp parallel num_threads(THREAD_NUM) shared(a) private(i, j, k, t1, t2)
	for (int k = 0; k < n; k++)
	{
		t1 = a[k][k];
#pragma omp single
		for (int j = k + 1; j < n; j++)
		{
			a[k][j] = a[k][j] / t1;
		}
		a[k][k] = 1.0;
#pragma omp for schedule(dynamic, chunksize)
		for (int i = k + 1; i < n; i++)
		{
			t2 = a[i][k];
			for (int j = k + 1; j < n; j++)
			{
				//cout << omp_get_thread_num() << endl;
				//cout << "omp_get_num_threads " << omp_get_num_threads() << endl;
				a[i][j] -= t2 * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void C_GE_OMP_Static(float** a, int n) {  // 高斯消去算法的Cache优化版本 _OMP
	//__m128 va;
	float t1, t2;  // 使用两个浮点数暂存数据以减少程序中地址的访问次数
	int i, j, k;
#pragma omp parallel num_threads(THREAD_NUM) shared(a) private(i, j, k, t1, t2)
	for (int k = 0; k < n; k++)
	{
		t1 = a[k][k];
#pragma omp single
		for (int j = k + 1; j < n; j++)
		{
			a[k][j] = a[k][j] / t1;
		}
		a[k][k] = 1.0;
#pragma omp for schedule(static, chunksize)
		for (int i = k + 1; i < n; i++)
		{
			t2 = a[i][k];
			for (int j = k + 1; j < n; j++)
			{
				//cout << omp_get_thread_num() << endl;
				//cout << "omp_get_num_threads " << omp_get_num_threads() << endl;
				a[i][j] -= t2 * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void C_GE_OMP_Guided(float** a, int n) {  // 高斯消去算法的Cache优化版本 _OMP
	//__m128 va;
	float t1, t2;  // 使用两个浮点数暂存数据以减少程序中地址的访问次数
	int i, j, k;
#pragma omp parallel num_threads(THREAD_NUM) shared(a) private(i, j, k, t1, t2)
	for (int k = 0; k < n; k++)
	{
		t1 = a[k][k];
#pragma omp single
		for (int j = k + 1; j < n; j++)
		{
			a[k][j] = a[k][j] / t1;
		}
		a[k][k] = 1.0;
#pragma omp for schedule(guided, chunksize)
		for (int i = k + 1; i < n; i++)
		{
			t2 = a[i][k];
			for (int j = k + 1; j < n; j++)
			{
				//cout << omp_get_thread_num() << endl;
				//cout << "omp_get_num_threads " << omp_get_num_threads() << endl;
				a[i][j] -= t2 * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void OMP_Dynamic_Block_GE_ALL(float** a, int n) {  // 高斯消去算法的Cache优化版本 _OMP 块划分
	float t1, t2;  // 使用两个浮点数暂存数据以减少程序中地址的访问次数
	int i, j, k;
	int chunk_size = sqrt(n / THREAD_NUM);
#pragma omp parallel num_threads(THREAD_NUM) shared(a) private(i, j, k, t1, t2)
	for (int k = 0; k < n; k++)
	{
		t1 = a[k][k];
#pragma omp for schedule(dynamic, chunk_size)
		for (int j = k + 1; j < n; j++)
		{
			a[k][j] = a[k][j] / t1;
		}
		a[k][k] = 1.0;
#pragma omp for schedule(dynamic, chunk_size)
		for (int i = k + 1; i < n; i++)
		{
			t2 = a[i][k];
			for (int j = k + 1; j < n; j++)
			{
				a[i][j] -= t2 * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void OMP_Dynamic_Rotation_GE_ALL(float** a, int n) {  // 高斯消去算法的Cache优化版本 _OMP 循环划分
	float t1, t2;  // 使用两个浮点数暂存数据以减少程序中地址的访问次数
	int i, j, k;
#pragma omp parallel num_threads(THREAD_NUM) shared(a) private(i, j, k, t1, t2)

	for (int k = 0; k < n; k++)
	{
		t1 = a[k][k];
#pragma omp for schedule(dynamic)
		for (int j = k + 1; j < n; j++)
		{
			a[k][j] = a[k][j] / t1;
		}
		a[k][k] = 1.0;
#pragma omp for schedule(dynamic)
		for (int i = k + 1; i < n; i++)
		{
			t2 = a[i][k];
			for (int j = k + 1; j < n; j++)
			{
				a[i][j] -= t2 * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}


//----------------------------------------初始化----------------------------------------

float** generate(int n) {
	ifstream inp("input.txt");
	inp >> n;
	float** m = new float* [n];
	for (int i = 0; i < n; i++)
	{
		m[i] = new float[n];
		for (int j = 0; j < n; j++)
		{
			inp >> m[i][j];
		}
	}
	inp.close();
	return m;
}

float** aligned_generate(int n) {
	ifstream inp("input.txt");
	inp >> n;
	float** m = (float**)aligned_alloc(32 * n * sizeof(float**), 32);
	for (int i = 0; i < n; i++)
	{
		m[i] = (float*)aligned_alloc(32 * n * sizeof(float*), 32);
	}
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			inp >> m[i][j];
		}
	}
	inp.close();
	return m;
}

void global_generate(int n) {
	ifstream inp("input.txt");
	inp >> n;
	mat = new float* [n];
	for (int i = 0; i < n; i++)
	{
		mat[i] = new float[n];
		for (int j = 0; j < n; j++)
		{
			inp >> mat[i][j];
		}
	}
	inp.close();
}

//------------------------------------全局变量初始化------------------------------------

void reset(float** &a, int n)  // 检测程序正确性
{
	a = new float* [n];
	for (int i = 0; i < n; i++)
	{
		a[i] = new float[n];
		for (int j = 0; j < n; j++)
		{
			a[i][j] = rand() % 10;
		}
	}
}

void global_reset(int n)  // 检测程序正确性
{
	mat = new float* [n];
	for (int i = 0; i < n; i++)
	{
		mat[i] = new float[n];
		for (int j = 0; j < n; j++)
		{
			mat[i][j] = rand() % 10;
		}
	}
}

//-------------------------------------展示所有内容-------------------------------------

void show(float** a, int n) {  // 用于观察程序运行结果
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << a[i][j] << " ";
			//out << a[i][j] << " ";
		}
		cout << endl;
		//out << endl;
	}

}

void show_in_file(float** a, int n) {  // 用于观察程序运行结果
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			//cout << a[i][j] << " ";
			out << a[i][j] << " ";
		}
		//cout << endl;
		out << endl;
	}

}

void sample_output(float** a, int n) {  // 用于观察程序运行结果
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			//cout << a[i][j] << " ";
			sampleout << a[i][j] << " ";
		}
		//cout << endl;
		sampleout << endl;
	}

}


int main() {
	srand(time(0));
	//in >> n;
	//cin >> n;
	n = 1000;
	out << n << endl;
	chunksize = 5;
	cout << "问题规模为" << n << "，算法的运行次数为" << lim << "，使用固定初始值, chunk_size = "<<chunksize<< endl;
	struct timeval start, end;
	float time_use = 0;
	//-----------------------------------------------------------------
	
	float** m2 = generate(n);

	//-----------------------------------------------------------------

	m2 = generate(n);
	gettimeofday(&start,NULL); //gettimeofday(&start,&tz);结果一样
	C_GE_OMP_Dynamic(m2, n);
    gettimeofday(&end,NULL); //gettimeofday(&start,&tz);结果一样
    time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);//微秒
	cout << "C_GE_OMP_Dynamic: " << time_use / 1000
		<< "ms" << endl;
	//show(m2, n);

	//-----------------------------------------------------------------

	m2 = generate(n);
	gettimeofday(&start,NULL); //gettimeofday(&start,&tz);结果一样
	C_GE_OMP_Static(m2, n);
    gettimeofday(&end,NULL); //gettimeofday(&start,&tz);结果一样
    time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);//微秒
	cout << "C_GE_OMP_Static: " << time_use / 1000
		<< "ms" << endl;
	//show(m2, n);

	//-----------------------------------------------------------------

	m2 = generate(n);
	gettimeofday(&start,NULL); //gettimeofday(&start,&tz);结果一样
	C_GE_OMP_Guided(m2, n);
    gettimeofday(&end,NULL); //gettimeofday(&start,&tz);结果一样
    time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);//微秒
	cout << "C_GE_OMP_Guided: " << time_use / 1000
		<< "ms" << endl;
	//show(m2, n);

	//-----------------------------------------------------------------
	/*
	m2 = generate(n);
	gettimeofday(&start,NULL); //gettimeofday(&start,&tz);结果一样
	OMP_Dynamic_Block_GE_ALL(m2, n);
    gettimeofday(&end,NULL); //gettimeofday(&start,&tz);结果一样
    time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);//微秒
	cout << "OMP_Dynamic_Block_GE_ALL: " << time_use / 1000
		<< "ms" << endl;
	//show(m2, n);

	//-----------------------------------------------------------------

	m2 = generate(n);
	gettimeofday(&start,NULL); //gettimeofday(&start,&tz);结果一样
	OMP_Dynamic_Rotation_GE_ALL(m2, n);
    gettimeofday(&end,NULL); //gettimeofday(&start,&tz);结果一样
    time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);//微秒
	cout << "OMP_Dynamic_Rotation_GE_ALL: " << time_use / 1000
		<< "ms" << endl;
	//show(m2, n);
	*/
	//-----------------------------------------------------------------
}
