#include<iostream>
#include<fstream>
#include<stdlib.h>
#include<ctime>
#include<cmath>

#include<windows.h>

#include<nmmintrin.h>  // SSE 4,2
#include<immintrin.h>  // AVX

#include<pthread.h>  // pthread
#include<semaphore.h>  // �ź���

#include<thread>  // C++11�������thread��
#include<mutex>
#include<condition_variable>

#include<omp.h>


using namespace std;

void reset(float**& a, int n);
void show(float** a, int n);
void GE(float** a, int n);
void C_GE(float** a, int n);
void SSE_GE(float** a, int n);
void AVX_GE(float** a, int n);

int n = 1000;
int lim = 1;
float** mat = nullptr;
const int THREAD_NUM = 7;  // �߳�����
ofstream out("output.txt");
ofstream sampleout("sampleout.txt");
//ifstream in("input.txt");

//------------------------------------�߳����ݽṹ------------------------------------

typedef struct {
	int k;  //��ȥ���ִ�
	int t_id;  // �߳� id
	int tasknum;  // ��������
}PT_EliminationParam;

typedef struct {
	int t_id; //�߳� id
}PT_StaticParam;

//-------------------------------------�ź�������-------------------------------------
sem_t sem_main;
sem_t sem_workerstart[THREAD_NUM]; // ÿ���߳����Լ�ר�����ź���
sem_t sem_workerend[THREAD_NUM];

sem_t sem_leader;
sem_t sem_Divsion[THREAD_NUM - 1];
sem_t sem_Elimination[THREAD_NUM - 1];

//------------------------------------barrier����-------------------------------------
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;

//-------------------------------���������ͻ���������---------------------------------
mutex _mutex;
condition_variable _cond;


//--------------------------------------�̺߳���--------------------------------------

void* PT_Block_Elimination(void* param) {
	PT_EliminationParam* tt = (PT_EliminationParam*)param;
	int k = tt->k;
	int t_id = tt->t_id;
	int tasknum = tt->tasknum;
	int i = k + t_id * tasknum + 1;
	float temp;  // ���˼��ͬC_GE
	if (t_id != THREAD_NUM - 1) {
		for (int c = 0; c < tasknum; i++, c++)  // ִ�б��̶߳�Ӧ������c�����������
		{
			temp = mat[i][k];
			for (int j = k + 1; j < n; j++)
			{
				mat[i][j] -= temp * mat[k][j];
			}
			mat[i][k] = 0;
		}
	}
	else {
		for (; i < n; i++)  // ִ�б��̶߳�Ӧ������c�����������
		{
			temp = mat[i][k];
			for (int j = k + 1; j < n; j++)
			{
				mat[i][j] -= temp * mat[k][j];
			}
			mat[i][k] = 0;
		}
	}
	pthread_exit(nullptr);
	return nullptr;
}

void* PT_Rotation_Elimination(void* param) {
	PT_EliminationParam* tt = (PT_EliminationParam*)param;
	int k = tt->k;
	int t_id = tt->t_id;
	float temp;  // ���˼��ͬC_GE
	for (int i = k + t_id + 1; i < n; i += THREAD_NUM)  // ִ�б��̶߳�Ӧ������c�����������
	{
		temp = mat[i][k];
		for (int j = k + 1; j < n; j++)
		{
			mat[i][j] -= temp * mat[k][j];
		}
		mat[i][k] = 0;
	}
	pthread_exit(nullptr);
	return nullptr;
}

void* PT_Static_Elimination(void* param) {
	PT_StaticParam* p = (PT_StaticParam*)param;
	int t_id = p->t_id;
	float t2;  // ʹ�ø������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	for (int k = 0; k < n; k++) {
		sem_wait(&sem_workerstart[t_id]); // �������ȴ�������ɳ��������������Լ�ר�����ź�����
		//ѭ����������
		for (int i = k + 1 + t_id; i < n; i += THREAD_NUM) {
			//��ȥ
			t2 = mat[i][k];
			for (int j = k + 1; j < n; j++) {
				mat[i][j] -= t2 * mat[k][j];
			}
			mat[i][k] = 0.0;
		}
		sem_post(&sem_main); // �������߳�
		sem_wait(&sem_workerend[t_id]); //�������ȴ����̻߳��ѽ�����һ��
	}
	pthread_exit(nullptr);
	return nullptr;
}

void* PT_Static_Div_Elem(void* param) {  // ����ѭ��ȫ������
	PT_StaticParam* p = (PT_StaticParam*)param;
	int t_id = p->t_id;
	float t1, t2;  // ʹ�ø������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	for (int k = 0; k < n; ++k) {
		// t_id Ϊ 0 ���߳����������������������߳��ȵȴ�
		// ����ֻ������һ�������̸߳������������ͬѧ�ǿ��Գ��Բ��ö�������߳���ɳ�������
		// ���ź���������ͬ����ʽ��ʹ�� barrier
		if (t_id == 0)
		{
			for (int j = k + 1; j < n; j++)
			{
				mat[k][j] = mat[k][j] / mat[k][k];
			}
			mat[k][k] = 1.0;
		}
		else {
			sem_wait(&sem_Divsion[t_id - 1]); // �������ȴ���ɳ�������
		}

		// t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
		if (t_id == 0) {
			for (int i = 0; i < THREAD_NUM - 1; i++) {
				sem_post(&sem_Divsion[i]);
			}
		}

		//ѭ����������ͬѧ�ǿ��Գ��Զ������񻮷ַ�ʽ��
		for (int i = k + 1 + t_id; i < n; i += THREAD_NUM) {
			//��ȥ
			for (int j = k + 1; j < n; j++) {
				mat[i][j] -= mat[i][k] * mat[k][j];
			}
			mat[i][k] = 0.0;
		}
		if (t_id == 0) {
			for (int i = 0; i < THREAD_NUM - 1; i++) {
				sem_wait(&sem_leader); // �ȴ����� worker �����ȥ
			}
			for (int i = 0; i < THREAD_NUM - 1; i++) {
				sem_post(&sem_Elimination[i]); // ֪ͨ���� worker ������һ��
			}
		}
		else {
			sem_post(&sem_leader);// ֪ͨ leader, �������ȥ����
			sem_wait(&sem_Elimination[t_id - 1]); // �ȴ�֪ͨ��������һ��
		}
	}
	pthread_exit(nullptr);
	return nullptr;
}

void* PT_Static_Barrier_Div_Elem(void* param) {
	PT_StaticParam* p = (PT_StaticParam*)param;
	int t_id = p->t_id;
	float t1, t2;  // ʹ�ø������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	for (int k = 0; k < n; ++k) {
		// t_id Ϊ 0 ���߳����������������������߳��ȵȴ�
		// ����ֻ������һ�������̸߳������������ͬѧ�ǿ��Գ��Բ��ö�������߳���ɳ�������
		if (t_id == 0) {
			t1 = mat[k][k];
			for (int j = k + 1; j < n; j++) {
				mat[k][j] /= t1;
			}
			mat[k][k] = 1.0;
		}
		//��һ��ͬ����
		pthread_barrier_wait(&barrier_Divsion);
		//ѭ����������ͬѧ�ǿ��Գ��Զ������񻮷ַ�ʽ��
		for (int i = k + 1 + t_id; i < n; i += THREAD_NUM) {
			//��ȥ
			t2 = mat[i][k];
			for (int j = k + 1; j < n; ++j) {
				mat[i][j] -= t2 * mat[k][j];
			}
			mat[i][k] = 0.0;
		}
		// �ڶ���ͬ����
		pthread_barrier_wait(&barrier_Elimination);
	}
	pthread_exit(nullptr);
	return nullptr;
}

void* PT_Static_Barrier_Div_Elem_Block(void* param) {
	PT_StaticParam* p = (PT_StaticParam*)param;
	int t_id = p->t_id;
	float t1, t2;  // ʹ�ø������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	int tasknum = 0;
	for (int k = 0; k < n; ++k) {
		// t_id Ϊ 0 ���߳����������������������߳��ȵȴ�
		// ����ֻ������һ�������̸߳������������ͬѧ�ǿ��Գ��Բ��ö�������߳���ɳ�������
		if (t_id == 0) {
			t1 = mat[k][k];
			for (int j = k + 1; j < n; j++) {
				mat[k][j] /= t1;
			}
			mat[k][k] = 1.0;
		}
		//��һ��ͬ����
		pthread_barrier_wait(&barrier_Divsion);
		//���ÿ黮��
		tasknum = (n - k - 1) / (THREAD_NUM - 1);
		int i = k + t_id * tasknum + 1;
		if (t_id != THREAD_NUM - 1) {
			for (int c = 0; c < tasknum; i++, c++)  // ִ�б��̶߳�Ӧ������c�����������
			{
				t2 = mat[i][k];
				for (int j = k + 1; j < n; j++)
				{
					mat[i][j] -= t2 * mat[k][j];
				}
				mat[i][k] = 0;
			}
		}
		else {
			for (; i < n; i++)  // ִ�б��̶߳�Ӧ������c�����������
			{
				t2 = mat[i][k];
				for (int j = k + 1; j < n; j++)
				{
					mat[i][j] -= t2 * mat[k][j];
				}
				mat[i][k] = 0;
			}
		}
		// �ڶ���ͬ����
		pthread_barrier_wait(&barrier_Elimination);
	}
	pthread_exit(nullptr);
	return nullptr;
}

//--------------------------------ʹ��AVX256----SIMD��--------------------------------

void* PT_Block_Elimination_AVX(void* param) {
	PT_EliminationParam* tt = (PT_EliminationParam*)param;
	int k = tt->k;
	int t_id = tt->t_id;
	int tasknum = tt->tasknum;
	int i = k + t_id * tasknum + 1;
	float temp;  // ���˼��ͬC_GE
	__m256 vaik, vakj, vaij, vx;
	int j;
	if (t_id != THREAD_NUM - 1) {
		for (int c = 0; c < tasknum; i++, c++)  // ִ�б��̶߳�Ӧ������c�����������
		{
			vaik = _mm256_set1_ps(mat[i][k]);
			temp = mat[i][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_loadu_ps(&mat[k][j]);
				vaij = _mm256_loadu_ps(&mat[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&mat[i][j], vaij);
			}
			for (j; j < n; j++)
			{
				mat[i][j] -= temp * mat[k][j];
			}
			mat[i][k] = 0;
		}
	}
	else {
		for (; i < n; i++)  // ִ�б��̶߳�Ӧ������c�����������
		{
			vaik = _mm256_set1_ps(mat[i][k]);
			temp = mat[i][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_loadu_ps(&mat[k][j]);
				vaij = _mm256_loadu_ps(&mat[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&mat[i][j], vaij);
			}
			for (j; j < n; j++)
			{
				mat[i][j] -= temp * mat[k][j];
			}
			mat[i][k] = 0;
		}
	}
	pthread_exit(nullptr);
	return nullptr;
}

void* PT_Rotation_Elimination_AVX(void* param) {
	PT_EliminationParam* tt = (PT_EliminationParam*)param;
	int k = tt->k;
	int t_id = tt->t_id;
	float temp;  // ���˼��ͬC_GE
	__m256 vaik, vakj, vaij, vx;
	int j = 0;
	for (int i = k + t_id + 1; i < n; i += THREAD_NUM)  // ִ�б��̶߳�Ӧ������c�����������
	{
		vaik = _mm256_set1_ps(mat[i][k]);
		temp = mat[i][k];
		for (j = k + 1; j + 8 < n; j += 8)
		{
			vakj = _mm256_loadu_ps(&mat[k][j]);
			vaij = _mm256_loadu_ps(&mat[i][j]);
			vx = _mm256_mul_ps(vakj, vaik);
			vaij = _mm256_sub_ps(vaij, vx);
			_mm256_storeu_ps(&mat[i][j], vaij);
		}
		for (j; j < n; j++)
		{
			mat[i][j] -= temp * mat[k][j];
		}
		mat[i][k] = 0;
	}
	pthread_exit(nullptr);
	return nullptr;
}

void* PT_Static_Elimination_AVX(void* param) {
	PT_StaticParam* p = (PT_StaticParam*)param;
	int t_id = p->t_id;
	float t2;  // ʹ�ø������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	__m256 vaik, vakj, vaij, vx;
	int j = 0;
	for (int k = 0; k < n; k++) {
		sem_wait(&sem_workerstart[t_id]); // �������ȴ�������ɳ��������������Լ�ר�����ź�����
		//ѭ����������
		for (int i = k + 1 + t_id; i < n; i += THREAD_NUM) {
			//��ȥ
			vaik = _mm256_set1_ps(mat[i][k]);
			t2 = mat[i][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_loadu_ps(&mat[k][j]);
				vaij = _mm256_loadu_ps(&mat[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&mat[i][j], vaij);
			}
			for (j; j < n; j++)
			{
				mat[i][j] -= t2 * mat[k][j];
			}
			mat[i][k] = 0;
		}
		sem_post(&sem_main); // �������߳�
		sem_wait(&sem_workerend[t_id]); //�������ȴ����̻߳��ѽ�����һ��
	}
	pthread_exit(nullptr);
	return nullptr;
}

void* PT_Static_Div_Elem_AVX(void* param) {  // ����ѭ��ȫ������
	PT_StaticParam* p = (PT_StaticParam*)param;
	int t_id = p->t_id;
	float t1, t2;  // ʹ�ø������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	__m256 va, vt, vaik, vakj, vaij, vx;
	int j = 0;
	for (int k = 0; k < n; ++k) {
		// t_id Ϊ 0 ���߳����������������������߳��ȵȴ�
		// ����ֻ������һ�������̸߳������������ͬѧ�ǿ��Գ��Բ��ö�������߳���ɳ�������
		// ���ź���������ͬ����ʽ��ʹ�� barrier
		if (t_id == 0)
		{
			vt = _mm256_set1_ps(mat[k][k]);  // �Գ����㷨����SIMD���л�
			t1 = mat[k][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				va = _mm256_loadu_ps(&mat[k][j]);
				va = _mm256_div_ps(va, vt);
				_mm256_storeu_ps(&mat[k][j], va);
			}
			for (j; j < n; j++)
			{
				mat[k][j] = mat[k][j] / t1;  // �ƺ�
			}
			mat[k][k] = 1.0;
		}
		else {
			sem_wait(&sem_Divsion[t_id - 1]); // �������ȴ���ɳ�������
		}

		// t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
		if (t_id == 0) {
			for (int i = 0; i < THREAD_NUM - 1; i++) {
				sem_post(&sem_Divsion[i]);
			}
		}

		//ѭ����������ͬѧ�ǿ��Գ��Զ������񻮷ַ�ʽ��
		for (int i = k + 1 + t_id; i < n; i += THREAD_NUM) {
			//��ȥ
			vaik = _mm256_set1_ps(mat[i][k]);
			t2 = mat[i][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_loadu_ps(&mat[k][j]);
				vaij = _mm256_loadu_ps(&mat[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&mat[i][j], vaij);
			}
			for (j; j < n; j++)
			{
				mat[i][j] -= t2 * mat[k][j];
			}
			mat[i][k] = 0;
		}
		if (t_id == 0) {
			for (int i = 0; i < THREAD_NUM - 1; i++) {
				sem_wait(&sem_leader); // �ȴ����� worker �����ȥ
			}
			for (int i = 0; i < THREAD_NUM - 1; i++) {
				sem_post(&sem_Elimination[i]); // ֪ͨ���� worker ������һ��
			}
		}
		else {
			sem_post(&sem_leader);// ֪ͨ leader, �������ȥ����
			sem_wait(&sem_Elimination[t_id - 1]); // �ȴ�֪ͨ��������һ��
		}
	}
	pthread_exit(nullptr);
	return nullptr;
}

void* PT_Static_Barrier_Div_Elem_AVX(void* param) {
	PT_StaticParam* p = (PT_StaticParam*)param;
	int t_id = p->t_id;
	float t1, t2;  // ʹ�ø������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	__m256 va, vt, vaik, vakj, vaij, vx;
	int j = 0;
	for (int k = 0; k < n; ++k) {
		// t_id Ϊ 0 ���߳����������������������߳��ȵȴ�
		// ����ֻ������һ�������̸߳������������ͬѧ�ǿ��Գ��Բ��ö�������߳���ɳ�������
		if (t_id == 0) {
			vt = _mm256_set1_ps(mat[k][k]);  // �Գ����㷨����SIMD���л�
			t1 = mat[k][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				va = _mm256_loadu_ps(&mat[k][j]);
				va = _mm256_div_ps(va, vt);
				_mm256_storeu_ps(&mat[k][j], va);
			}
			for (j; j < n; j++)
			{
				mat[k][j] = mat[k][j] / t1;  // �ƺ�
			}
			mat[k][k] = 1.0;
		}
		//��һ��ͬ����
		pthread_barrier_wait(&barrier_Divsion);
		//ѭ����������ͬѧ�ǿ��Գ��Զ������񻮷ַ�ʽ��
		for (int i = k + 1 + t_id; i < n; i += THREAD_NUM) {
			//��ȥ
			vaik = _mm256_set1_ps(mat[i][k]);
			t2 = mat[i][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_loadu_ps(&mat[k][j]);
				vaij = _mm256_loadu_ps(&mat[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&mat[i][j], vaij);
			}
			for (j; j < n; j++)
			{
				mat[i][j] -= t2 * mat[k][j];
			}
			mat[i][k] = 0;
		}
		// �ڶ���ͬ����
		pthread_barrier_wait(&barrier_Elimination);
	}
	pthread_exit(nullptr);
	return nullptr;
}

void* PT_Static_Barrier_Div_Elem_Block_AVX(void* param) {
	PT_StaticParam* p = (PT_StaticParam*)param;
	int t_id = p->t_id;
	float t1, t2;  // ʹ�ø������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	int tasknum = 0;
	int j = 0;
	__m256 va, vt, vaik, vakj, vaij, vx;
	for (int k = 0; k < n; ++k) {
		// t_id Ϊ 0 ���߳����������������������߳��ȵȴ�
		// ����ֻ������һ�������̸߳������������ͬѧ�ǿ��Գ��Բ��ö�������߳���ɳ�������
		if (t_id == 0) {
			vt = _mm256_set1_ps(mat[k][k]);  // �Գ����㷨����SIMD���л�
			t1 = mat[k][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				va = _mm256_loadu_ps(&mat[k][j]);
				va = _mm256_div_ps(va, vt);
				_mm256_storeu_ps(&mat[k][j], va);
			}
			for (j; j < n; j++)
			{
				mat[k][j] = mat[k][j] / t1;  // �ƺ�
			}
			mat[k][k] = 1.0;
		}
		//��һ��ͬ����
		pthread_barrier_wait(&barrier_Divsion);
		//���ÿ黮��
		tasknum = (n - k - 1) / (THREAD_NUM - 1);
		int i = k + t_id * tasknum + 1;
		if (t_id != THREAD_NUM - 1) {
			for (int c = 0; c < tasknum; i++, c++)  // ִ�б��̶߳�Ӧ������c�����������
			{
				//��ȥ
				vaik = _mm256_set1_ps(mat[i][k]);
				t2 = mat[i][k];
				for (j = k + 1; j + 8 < n; j += 8)
				{
					vakj = _mm256_loadu_ps(&mat[k][j]);
					vaij = _mm256_loadu_ps(&mat[i][j]);
					vx = _mm256_mul_ps(vakj, vaik);
					vaij = _mm256_sub_ps(vaij, vx);
					_mm256_storeu_ps(&mat[i][j], vaij);
				}
				for (j; j < n; j++)
				{
					mat[i][j] -= t2 * mat[k][j];
				}
				mat[i][k] = 0;
			}
		}
		else {
			for (; i < n; i++)  // ִ�б��̶߳�Ӧ������c�����������
			{
				//��ȥ
				vaik = _mm256_set1_ps(mat[i][k]);
				t2 = mat[i][k];
				for (j = k + 1; j + 8 < n; j += 8)
				{
					vakj = _mm256_loadu_ps(&mat[k][j]);
					vaij = _mm256_loadu_ps(&mat[i][j]);
					vx = _mm256_mul_ps(vakj, vaik);
					vaij = _mm256_sub_ps(vaij, vx);
					_mm256_storeu_ps(&mat[i][j], vaij);
				}
				for (j; j < n; j++)
				{
					mat[i][j] -= t2 * mat[k][j];
				}
				mat[i][k] = 0;
			}
		}
		// �ڶ���ͬ����
		pthread_barrier_wait(&barrier_Elimination);
	}
	pthread_exit(nullptr);
	return nullptr;
}

//--------------------------------------��ȥ�㷨--------------------------------------

void GE(float** a, int n) {  // ��׼�ĸ�˹��ȥ�㷨, Gauss Elimination��д
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


void C_GE(float** a, int n) {  // ��˹��ȥ�㷨��Cache�Ż��汾
	//__m128 va;
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
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


//--------------------------------------��ȥ�㷨(OMP)--------------------------------------

void GE_OMP(float** a, int n) {  // ��׼�ĸ�˹��ȥ�㷨, Gauss Elimination��д _OMP
	int i, j, k;
#pragma omp parallel num_threads(THREAD_NUM) shared(a) private(i, j, k)
	{
		for (int k = 0; k < n; k++)
		{
#pragma omp single
			for (int j = k + 1; j < n; j++)
			{
				{
				
				a[k][j] = a[k][j] / a[k][k];
				//cout << omp_get_thread_num() << endl;
				//cout <<"omp_get_num_threads "<< omp_get_num_threads() << endl;
				}
			}
			a[k][k] = 1.0;

#pragma omp for 
			for (int i = k + 1; i < n; i++)
			{
				for (int j = k + 1; j < n; j++)
				{
					a[i][j] -= a[i][k] * a[k][j];
					//cout << omp_get_thread_num() << endl;
					//cout << "omp_get_num_threads " << omp_get_num_threads() << endl;
				}
				a[i][k] = 0;
			}
		}
	}
}

void C_GE_OMP(float** a, int n) {  // ��˹��ȥ�㷨��Cache�Ż��汾 _OMP
	//__m128 va;
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
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
#pragma omp for 
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

void C_GE_OMP_Dynamic(float** a, int n) {  // ��˹��ȥ�㷨��Cache�Ż��汾 _OMP
	//__m128 va;
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
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
#pragma omp for schedule(dynamic)
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

void C_GE_OMP_Static(float** a, int n) {  // ��˹��ȥ�㷨��Cache�Ż��汾 _OMP
	//__m128 va;
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
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
#pragma omp for schedule(static)
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

void C_GE_OMP_Guided(float** a, int n) {  // ��˹��ȥ�㷨��Cache�Ż��汾 _OMP
	//__m128 va;
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
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
#pragma omp for schedule(guided)
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

void OMP_Dynamic_Block_GE_ALL(float** a, int n) {  // ��˹��ȥ�㷨��Cache�Ż��汾 _OMP �黮��
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
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

void OMP_Dynamic_Rotation_GE_ALL(float** a, int n) {  // ��˹��ȥ�㷨��Cache�Ż��汾 _OMP ѭ������
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
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


void C_GE_OMP_Dynamic_AVX(float** a, int n) {  // ʹ��AVXָ�����SIMD�Ż��ĸ�˹��ȥ�㷨
	__m256 va, vt, vaik, vakj, vaij, vx;
	int i, j, k;
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
#pragma omp parallel num_threads(THREAD_NUM) shared(a) private(i, j, k, t1, t2, va, vt, vaik, vakj, vaij, vx)
	for (int k = 0; k < n; k++)
	{
		vt = _mm256_set1_ps(a[k][k]);
		t1 = a[k][k];
#pragma omp single
		for (j = k + 1; j + 8 < n; j += 8)
		{
			va = _mm256_loadu_ps(&a[k][j]);
			va = _mm256_div_ps(va, vt);
			_mm256_storeu_ps(&a[k][j], va);
		}
#pragma omp single
		for (j; j < n; j++)
		{
			a[k][j] = a[k][j] / t1;  // �ƺ�
		}
		a[k][k] = 1.0;
#pragma omp for schedule(dynamic)
		for (int i = k + 1; i < n; i++)
		{
			vaik = _mm256_set1_ps(a[i][k]);
			t2 = a[i][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_loadu_ps(&a[k][j]);
				vaij = _mm256_loadu_ps(&a[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&a[i][j], vaij);
			}
			for (j; j < n; j++)
			{
				a[i][j] -= t2 * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void C_GE_OMP_Static_AVX(float** a, int n) {  // ʹ��AVXָ�����SIMD�Ż��ĸ�˹��ȥ�㷨
	__m256 va, vt, vaik, vakj, vaij, vx;
	int i, j, k;
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
#pragma omp parallel num_threads(THREAD_NUM) shared(a) private(i, j, k, t1, t2, va, vt, vaik, vakj, vaij, vx)
	for (int k = 0; k < n; k++)
	{
		vt = _mm256_set1_ps(a[k][k]);
		t1 = a[k][k];
#pragma omp single
		for (j = k + 1; j + 8 < n; j += 8)
		{
			va = _mm256_loadu_ps(&a[k][j]);
			va = _mm256_div_ps(va, vt);
			_mm256_storeu_ps(&a[k][j], va);
		}
#pragma omp single
		for (j; j < n; j++)
		{
			a[k][j] = a[k][j] / t1;  // �ƺ�
		}
		a[k][k] = 1.0;
#pragma omp for schedule(static)
		for (int i = k + 1; i < n; i++)
		{
			vaik = _mm256_set1_ps(a[i][k]);
			t2 = a[i][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_loadu_ps(&a[k][j]);
				vaij = _mm256_loadu_ps(&a[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&a[i][j], vaij);
			}
			for (j; j < n; j++)
			{
				a[i][j] -= t2 * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void C_GE_OMP_Guided_AVX(float** a, int n) {  // ʹ��AVXָ�����SIMD�Ż��ĸ�˹��ȥ�㷨
	__m256 va, vt, vaik, vakj, vaij, vx;
	int i, j, k;
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
#pragma omp parallel num_threads(THREAD_NUM) shared(a) private(i, j, k, t1, t2, va, vt, vaik, vakj, vaij, vx)
	for (int k = 0; k < n; k++)
	{
		vt = _mm256_set1_ps(a[k][k]);
		t1 = a[k][k];
#pragma omp single
		for (j = k + 1; j + 8 < n; j += 8)
		{
			va = _mm256_loadu_ps(&a[k][j]);
			va = _mm256_div_ps(va, vt);
			_mm256_storeu_ps(&a[k][j], va);
		}
#pragma omp single
		for (j; j < n; j++)
		{
			a[k][j] = a[k][j] / t1;  // �ƺ�
		}
		a[k][k] = 1.0;
#pragma omp for schedule(guided)
		for (int i = k + 1; i < n; i++)
		{
			vaik = _mm256_set1_ps(a[i][k]);
			t2 = a[i][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_loadu_ps(&a[k][j]);
				vaij = _mm256_loadu_ps(&a[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&a[i][j], vaij);
			}
			for (j; j < n; j++)
			{
				a[i][j] -= t2 * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void C_GE_OMP_Dynamic_AUTO_SIMD(float** a, int n) {  // ��˹��ȥ�㷨��Cache�Ż��汾 _OMP
	//__m128 va;
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	int i, j, k;
#pragma omp parallel num_threads(THREAD_NUM) shared(a) private(i, j, k, t1, t2)
	for (int k = 0; k < n; k++)
	{
		t1 = a[k][k];
#pragma omp simd 
		for (int j = k + 1; j < n; j++)
		{
			a[k][j] = a[k][j] / t1;
		}
#pragma omp single
		a[k][k] = 1.0;
#pragma omp for schedule(dynamic)
		for (int i = k + 1; i < n; i++)
		{
			t2 = a[i][k];
#pragma omp simd 
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

/*
void OMP_Dynamic_Block_GE_ALL_AVX(float** a, int n) {  // ʹ��AVXָ�����SIMD�Ż��ĸ�˹��ȥ�㷨
	__m256 va, vt, vaik, vakj, vaij, vx;
	int i, j, k;
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	int chunk_size = sqrt(n / THREAD_NUM);
#pragma omp parallel num_threads(THREAD_NUM) shared(a) private(i, j, k, t1, t2, va, vt, vaik, vakj, vaij, vx)
	for (int k = 0; k < n; k++)
	{
		vt = _mm256_set1_ps(a[k][k]);
		t1 = a[k][k];
#pragma omp for schedule(dynamic, chunk_size)
		for (j = k + 1; j + 8 < n; j += 8)
		{
			va = _mm256_loadu_ps(&a[k][j]);
			va = _mm256_div_ps(va, vt);
			_mm256_storeu_ps(&a[k][j], va);
		}
#pragma omp single
		for (j; j < n; j++)
		{
			a[k][j] = a[k][j] / t1;  // �ƺ�
		}
		a[k][k] = 1.0;
#pragma omp for schedule(dynamic, chunk_size)
		for (int i = k + 1; i < n; i++)
		{
			vaik = _mm256_set1_ps(a[i][k]);
			t2 = a[i][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_loadu_ps(&a[k][j]);
				vaij = _mm256_loadu_ps(&a[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&a[i][j], vaij);
			}
			for (j; j < n; j++)
			{
				a[i][j] -= t2 * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void OMP_Dynamic_Rotation_GE_ALL_AVX(float** a, int n) {  // ʹ��AVXָ�����SIMD�Ż��ĸ�˹��ȥ�㷨
	__m256 va, vt, vaik, vakj, vaij, vx;
	int i, j, k;
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
#pragma omp parallel num_threads(THREAD_NUM) shared(a) private(i, j, k, t1, t2, va, vt, vaik, vakj, vaij, vx)
	for (int k = 0; k < n; k++)
	{
		vt = _mm256_set1_ps(a[k][k]);
		t1 = a[k][k];
#pragma omp for schedule(dynamic, 1)
		for (j = k + 1; j + 8 < n; j += 8)
		{
			va = _mm256_loadu_ps(&a[k][j]);
			va = _mm256_div_ps(va, vt);
			_mm256_storeu_ps(&a[k][j], va);
		}
#pragma omp single
		for (j; j < n; j++)
		{
			a[k][j] = a[k][j] / t1;  // �ƺ�
		}
		a[k][k] = 1.0;
#pragma omp for schedule(dynamic, 1)
		for (int i = k + 1; i < n; i++)
		{
			vaik = _mm256_set1_ps(a[i][k]);
			t2 = a[i][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_loadu_ps(&a[k][j]);
				vaij = _mm256_loadu_ps(&a[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&a[i][j], vaij);
			}
			for (j; j < n; j++)
			{
				a[i][j] -= t2 * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

*/



//-----------------------------------PThread��ȥ�㷨----------------------------------

void PThread_Dynamic_Block_GE(int n) {  // PThread�Ż��ĸ�˹��ȥ�㷨����̬�����黮������
	int tasknum = 0;
	pthread_t* handles = (pthread_t*)malloc((THREAD_NUM) * sizeof(pthread_t));  // Ϊ�߳̾�������ڴ�ռ�
	PT_EliminationParam* param = (PT_EliminationParam*)malloc((THREAD_NUM) * sizeof(PT_EliminationParam));  // �洢�̲߳���
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	for (int k = 0; k < n; k++)
	{
		t1 = mat[k][k];
		for (int j = k + 1; j < n; j++)
		{
			mat[k][j] = mat[k][j] / t1;
		}
		mat[k][k] = 1.0;
		//int thread_count = n - 1 - k;  // �����߳�����
		tasknum = (n - k - 1) / (THREAD_NUM - 1);  // ���߳���ִ�е����������������һ��Ϊ��Ϊ���ʣ�µ����������ռ�
		//��������
		if (tasknum < THREAD_NUM)  // ���һ���̵߳����������������߳������࣬��ô��ֱ�Ӳ������߳�
		{
			for (int i = k + 1; i < n; i++)
			{
				t2 = mat[i][k];
				for (int j = k + 1; j < n; j++)
				{
					mat[i][j] -= t2 * mat[k][j];
				}
				mat[i][k] = 0;
			}
		}
		else {  // ���򴴽����߳�
			for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
				//cout << k << " " << t_id << endl;
				param[t_id].k = k;
				param[t_id].t_id = t_id;
				param[t_id].tasknum = tasknum;
			}
			//�����߳�
			for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
				pthread_create(&handles[t_id], nullptr, PT_Block_Elimination, (void*)&param[t_id]);
			}
			for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
				pthread_join(handles[t_id], nullptr);
			}
			//cout << endl << tasknum << " " << thread_count << " " << tasknum * thread_count << " " << n - 1 - k << endl;
		}
	}
}

void PThread_Dynamic_Rotation_GE(int n) {  // PThread�Ż��ĸ�˹��ȥ�㷨����̬��������������
	int tasknum = 0;
	pthread_t* handles = (pthread_t*)malloc(THREAD_NUM * sizeof(pthread_t));  // Ϊ�߳̾�������ڴ�ռ�
	PT_EliminationParam* param = (PT_EliminationParam*)malloc(THREAD_NUM * sizeof(PT_EliminationParam));  // �洢�̲߳���
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	for (int k = 0; k < n; k++)
	{
		t1 = mat[k][k];
		for (int j = k + 1; j < n; j++)
		{
			mat[k][j] = mat[k][j] / t1;
		}
		mat[k][k] = 1.0;
		//��������
		tasknum = (n - k - 1) / (THREAD_NUM);  // ���߳���ִ�е����������������һ��Ϊ��Ϊ���ʣ�µ����������ռ�
		if (tasknum < THREAD_NUM)  // ���һ���̵߳����������������߳������࣬��ô��ֱ�Ӳ������߳�
		{
			for (int i = k + 1; i < n; i++)
			{
				t2 = mat[i][k];
				for (int j = k + 1; j < n; j++)
				{
					mat[i][j] -= t2 * mat[k][j];
				}
				mat[i][k] = 0;
			}
		}
		else {
			for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
				//cout << k << " " << t_id << endl;
				param[t_id].k = k;
				param[t_id].t_id = t_id;
			}
			//�����߳�
			for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
				pthread_create(&handles[t_id], nullptr, PT_Rotation_Elimination, (void*)&param[t_id]);
			}
			for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
				pthread_join(handles[t_id], nullptr);
			}
		}
		//cout << endl << tasknum << " " << thread_count << " " << tasknum * thread_count << " " << n - 1 - k << endl;
	}
}

void PThread_Static_OnlyElim_GE(int n) {  // PThread�Ż��ĸ�˹��ȥ�㷨����̬��ֻ������ѭ���еĺ���������߳�
	//��ʼ���ź���
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < THREAD_NUM; i++) {
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}
	//�����߳�
	pthread_t handles[THREAD_NUM];// ������Ӧ�� Handle
	PT_StaticParam param[THREAD_NUM];// ������Ӧ���߳����ݽṹ
	float t1;  // ʹ�ø������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], nullptr, PT_Static_Elimination, (void*)&param[t_id]);
	}
	for (int k = 0; k < n; k++) {
		//���߳�����������
		t1 = mat[k][k];
		for (int j = k + 1; j < n; j++) {
			mat[k][j] /= t1;
		}
		mat[k][k] = 1.0;
		//��ʼ���ѹ����߳�
		for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
			sem_post(&sem_workerstart[t_id]);
		}
		//���߳�˯�ߣ��ȴ����еĹ����߳���ɴ�����ȥ����
		for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
			sem_wait(&sem_main);
		}
		// ���߳��ٴλ��ѹ����߳̽�����һ�ִε���ȥ����
		for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
			sem_post(&sem_workerend[t_id]);
		}
	}
	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		pthread_join(handles[t_id], nullptr);
	}
	//���������ź���
	sem_destroy(&sem_main);
	sem_destroy(sem_workerstart);
	sem_destroy(sem_workerend);
}

void PThread_Static_GE(int n) {  // PThread�Ż��ĸ�˹��ȥ�㷨����̬��������ѭ��ȫ�������߳�
	//��ʼ���ź���
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < THREAD_NUM - 1; ++i) {
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
	//�����߳�
	pthread_t handles[THREAD_NUM];// ������Ӧ�� Handle
	PT_StaticParam param[THREAD_NUM];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], nullptr, PT_Static_Div_Elem, (void*)&param[t_id]);
	}
	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		pthread_join(handles[t_id], nullptr);
	}
	// ���������ź���
	sem_destroy(&sem_leader);
	sem_destroy(sem_Divsion);
	sem_destroy(sem_Elimination);
}

void PThread_Static_Barrier_GE(int n) {
	//��ʼ�� barrier
	pthread_barrier_init(&barrier_Divsion, nullptr, THREAD_NUM);
	pthread_barrier_init(&barrier_Elimination, nullptr, THREAD_NUM);
	//�����߳�
	pthread_t handles[THREAD_NUM];// ������Ӧ�� Handle
	PT_StaticParam param[THREAD_NUM];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], nullptr, PT_Static_Barrier_Div_Elem, (void*)&param[t_id]);
	}

	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		pthread_join(handles[t_id], nullptr);
	}
	//�������е� barrier
	pthread_barrier_destroy(&barrier_Divsion);
	pthread_barrier_destroy(&barrier_Elimination);
}

void PThread_Static_Barrier_Block_GE(int n) {
	//��ʼ�� barrier
	pthread_barrier_init(&barrier_Divsion, nullptr, THREAD_NUM);
	pthread_barrier_init(&barrier_Elimination, nullptr, THREAD_NUM);
	//�����߳�
	pthread_t handles[THREAD_NUM];// ������Ӧ�� Handle
	PT_StaticParam param[THREAD_NUM];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], nullptr, PT_Static_Barrier_Div_Elem_Block, (void*)&param[t_id]);
	}

	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		pthread_join(handles[t_id], nullptr);
	}
	//�������е� barrier
	pthread_barrier_destroy(&barrier_Divsion);
	pthread_barrier_destroy(&barrier_Elimination);
}

//-------------------------------------����SIMD�㷨------------------------------------

void AVX_GE(float** a, int n) {  // ʹ��AVXָ�����SIMD�Ż��ĸ�˹��ȥ�㷨
	__m256 va, vt, vaik, vakj, vaij, vx;
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	for (int k = 0; k < n; k++)
	{
		vt = _mm256_set1_ps(a[k][k]);
		t1 = a[k][k];
		int j = 0;
		for (j = k + 1; j + 8 < n; j += 8)
		{
			va = _mm256_loadu_ps(&a[k][j]);
			va = _mm256_div_ps(va, vt);
			_mm256_storeu_ps(&a[k][j], va);
		}
		for (j; j < n; j++)
		{
			a[k][j] = a[k][j] / t1;  // �ƺ�
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik = _mm256_set1_ps(a[i][k]);
			t2 = a[i][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_loadu_ps(&a[k][j]);
				vaij = _mm256_loadu_ps(&a[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&a[i][j], vaij);
			}
			for (j; j < n; j++)
			{
				a[i][j] -= t2 * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

//--------------------------------PThread+SIMD��ȥ�㷨--------------------------------

void PThread_Dynamic_Block_GE_AVX(int n) {  // PThread�Ż��ĸ�˹��ȥ�㷨����̬�����黮������
	int tasknum = 0;
	pthread_t* handles = (pthread_t*)malloc((THREAD_NUM) * sizeof(pthread_t));  // Ϊ�߳̾�������ڴ�ռ�
	PT_EliminationParam* param = (PT_EliminationParam*)malloc((THREAD_NUM) * sizeof(PT_EliminationParam));  // �洢�̲߳���
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	__m256 va, vt, vaik, vakj, vaij, vx;
	for (int k = 0; k < n; k++)
	{
		vt = _mm256_set1_ps(mat[k][k]);  // ����ȥ�㷨����SIMD���л�
		t1 = mat[k][k];
		int j = 0;
		for (j = k + 1; j + 8 < n; j += 8)
		{
			va = _mm256_loadu_ps(&mat[k][j]);
			va = _mm256_div_ps(va, vt);
			_mm256_storeu_ps(&mat[k][j], va);
		}
		for (j; j < n; j++)
		{
			mat[k][j] = mat[k][j] / t1;  // �ƺ�
		}
		mat[k][k] = 1.0;

		tasknum = (n - k - 1) / (THREAD_NUM - 1);  // ���߳���ִ�е����������������һ��Ϊ��Ϊ���ʣ�µ����������ռ�
		//��������
		if (tasknum < THREAD_NUM)  // ���һ���̵߳����������������߳������࣬��ô��ֱ�Ӳ������߳�
		{
			for (int i = k + 1; i < n; i++)
			{
				vaik = _mm256_set1_ps(mat[i][k]);
				t2 = mat[i][k];
				for (j = k + 1; j + 8 < n; j += 8)
				{
					vakj = _mm256_loadu_ps(&mat[k][j]);
					vaij = _mm256_loadu_ps(&mat[i][j]);
					vx = _mm256_mul_ps(vakj, vaik);
					vaij = _mm256_sub_ps(vaij, vx);
					_mm256_storeu_ps(&mat[i][j], vaij);
				}
				for (j; j < n; j++)
				{
					mat[i][j] -= t2 * mat[k][j];
				}
				mat[i][k] = 0;
			}
		}
		else {  // ���򴴽����߳�
			for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
				//cout << k << " " << t_id << endl;
				param[t_id].k = k;
				param[t_id].t_id = t_id;
				param[t_id].tasknum = tasknum;
			}
			//�����߳�
			for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
				pthread_create(&handles[t_id], nullptr, PT_Block_Elimination_AVX, (void*)&param[t_id]);
			}
			for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
				pthread_join(handles[t_id], nullptr);
			}
			//cout << endl << tasknum << " " << thread_count << " " << tasknum * thread_count << " " << n - 1 - k << endl;
		}
	}
}

void PThread_Dynamic_Rotation_GE_AVX(int n) {  // PThread�Ż��ĸ�˹��ȥ�㷨����̬��������������
	int tasknum = 0;
	pthread_t* handles = (pthread_t*)malloc(THREAD_NUM * sizeof(pthread_t));  // Ϊ�߳̾�������ڴ�ռ�
	PT_EliminationParam* param = (PT_EliminationParam*)malloc(THREAD_NUM * sizeof(PT_EliminationParam));  // �洢�̲߳���
	__m256 va, vt, vaik, vakj, vaij, vx;
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	for (int k = 0; k < n; k++)
	{
		vt = _mm256_set1_ps(mat[k][k]);
		t1 = mat[k][k];
		int j = 0;
		for (j = k + 1; j + 8 < n; j += 8)
		{
			va = _mm256_loadu_ps(&mat[k][j]);
			va = _mm256_div_ps(va, vt);
			_mm256_storeu_ps(&mat[k][j], va);
		}
		for (j; j < n; j++)
		{
			mat[k][j] = mat[k][j] / t1;  // �ƺ�
		}
		mat[k][k] = 1.0;
		//��������
		tasknum = (n - k - 1) / (THREAD_NUM);  // ���߳���ִ�е����������������һ��Ϊ��Ϊ���ʣ�µ����������ռ�
		if (tasknum < THREAD_NUM)  // ���һ���̵߳����������������߳������࣬��ô��ֱ�Ӳ������߳�
		{
			for (int i = k + 1; i < n; i++)
			{
				vaik = _mm256_set1_ps(mat[i][k]);
				t2 = mat[i][k];
				for (j = k + 1; j + 8 < n; j += 8)
				{
					vakj = _mm256_loadu_ps(&mat[k][j]);
					vaij = _mm256_loadu_ps(&mat[i][j]);
					vx = _mm256_mul_ps(vakj, vaik);
					vaij = _mm256_sub_ps(vaij, vx);
					_mm256_storeu_ps(&mat[i][j], vaij);
				}
				for (j; j < n; j++)
				{
					mat[i][j] -= t2 * mat[k][j];
				}
				mat[i][k] = 0;
			}
		}
		else {
			for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
				//cout << k << " " << t_id << endl;
				param[t_id].k = k;
				param[t_id].t_id = t_id;
			}
			//�����߳�
			for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
				pthread_create(&handles[t_id], nullptr, PT_Rotation_Elimination_AVX, (void*)&param[t_id]);
			}
			for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
				pthread_join(handles[t_id], nullptr);
			}
			//cout << endl << tasknum << " " << thread_count << " " << tasknum * thread_count << " " << n - 1 - k << endl;
		}
	}
}

void PThread_Static_OnlyElim_GE_AVX(int n) {  // PThread�Ż��ĸ�˹��ȥ�㷨����̬��ֻ������ѭ���еĺ���������߳�
	//��ʼ���ź���
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < THREAD_NUM; i++) {
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}
	//�����߳�
	pthread_t handles[THREAD_NUM];// ������Ӧ�� Handle
	PT_StaticParam param[THREAD_NUM];// ������Ӧ���߳����ݽṹ
	float t1;  // ʹ�ø������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	__m256 va, vt, vaik, vakj, vaij, vx;
	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], nullptr, PT_Static_Elimination_AVX, (void*)&param[t_id]);
	}
	for (int k = 0; k < n; k++) {
		//���߳�����������
		vt = _mm256_set1_ps(mat[k][k]);
		t1 = mat[k][k];
		int j = 0;
		for (j = k + 1; j + 8 < n; j += 8)
		{
			va = _mm256_loadu_ps(&mat[k][j]);
			va = _mm256_div_ps(va, vt);
			_mm256_storeu_ps(&mat[k][j], va);
		}
		for (j; j < n; j++)
		{
			mat[k][j] = mat[k][j] / t1;  // �ƺ�
		}
		mat[k][k] = 1.0;

		//��ʼ���ѹ����߳�
		for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
			sem_post(&sem_workerstart[t_id]);
		}
		//���߳�˯�ߣ��ȴ����еĹ����߳���ɴ�����ȥ����
		for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
			sem_wait(&sem_main);
		}
		// ���߳��ٴλ��ѹ����߳̽�����һ�ִε���ȥ����
		for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
			sem_post(&sem_workerend[t_id]);
		}
	}
	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		pthread_join(handles[t_id], nullptr);
	}
	//���������ź���
	sem_destroy(&sem_main);
	sem_destroy(sem_workerstart);
	sem_destroy(sem_workerend);
}

void PThread_Static_GE_AVX(int n) {  // PThread�Ż��ĸ�˹��ȥ�㷨����̬��������ѭ��ȫ�������߳�
	//��ʼ���ź���
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < THREAD_NUM - 1; ++i) {
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
	//�����߳�
	pthread_t handles[THREAD_NUM];// ������Ӧ�� Handle
	PT_StaticParam param[THREAD_NUM];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], nullptr, PT_Static_Div_Elem_AVX, (void*)&param[t_id]);
	}
	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		pthread_join(handles[t_id], nullptr);
	}
	// ���������ź���
	sem_destroy(&sem_leader);
	sem_destroy(sem_Divsion);
	sem_destroy(sem_Elimination);
}

void PThread_Static_Barrier_GE_AVX(int n) {
	//��ʼ�� barrier
	pthread_barrier_init(&barrier_Divsion, nullptr, THREAD_NUM);
	pthread_barrier_init(&barrier_Elimination, nullptr, THREAD_NUM);
	//�����߳�
	pthread_t handles[THREAD_NUM];// ������Ӧ�� Handle
	PT_StaticParam param[THREAD_NUM];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], nullptr, PT_Static_Barrier_Div_Elem_AVX, (void*)&param[t_id]);
	}

	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		pthread_join(handles[t_id], nullptr);
	}
	//�������е� barrier
	pthread_barrier_destroy(&barrier_Divsion);
	pthread_barrier_destroy(&barrier_Elimination);
}

void PThread_Static_Barrier_Block_GE_AVX(int n) {
	//��ʼ�� barrier
	pthread_barrier_init(&barrier_Divsion, nullptr, THREAD_NUM);
	pthread_barrier_init(&barrier_Elimination, nullptr, THREAD_NUM);
	//�����߳�
	pthread_t handles[THREAD_NUM];// ������Ӧ�� Handle
	PT_StaticParam param[THREAD_NUM];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], nullptr, PT_Static_Barrier_Div_Elem_Block_AVX, (void*)&param[t_id]);
	}

	for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
		pthread_join(handles[t_id], nullptr);
	}
	//�������е� barrier
	pthread_barrier_destroy(&barrier_Divsion);
	pthread_barrier_destroy(&barrier_Elimination);
}

//----------------------------------------��ʼ��----------------------------------------

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
	float** m = (float**)_aligned_malloc(32 * n * sizeof(float**), 32);
	for (int i = 0; i < n; i++)
	{
		m[i] = (float*)_aligned_malloc(32 * n * sizeof(float*), 32);
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

//------------------------------------ȫ�ֱ�����ʼ��------------------------------------

void reset(float**& a, int n)  // ��������ȷ��
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

void global_reset(int n)  // ��������ȷ��
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

//-------------------------------------չʾ��������-------------------------------------

void show(float** a, int n) {  // ���ڹ۲�������н��
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

void show_in_file(float** a, int n) {  // ���ڹ۲�������н��
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

void sample_output(float** a, int n) {  // ���ڹ۲�������н��
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
	cin >> n;
	out << n << endl;
	cout << "�����ģΪ" << n << "���㷨�����д���Ϊ" << lim << "��ʹ�ù̶���ʼֵ" << endl;
	long long head1, head2, head3, head4, head5, head6, tail1, tail2, tail3, tail4, tail5, tail6, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	//-----------------------------------------------------------------
	//float** m1 = new float* [n];
	float** m1 = generate(n);
	//reset(m1, n);

	//show(m1, n);
	show_in_file(m1, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head1);
	GE(m1, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail1);
	cout << endl << endl << endl;
	//show(m1, n);
	sample_output(m1, n);

	cout << "GE: " << (tail1 - head1) * 1000.0 / freq
		<< "ms" << endl;

	//-----------------------------------------------------------------

	//float** m1 = new float* [n];
	m1 = generate(n);
	//reset(m1, n);

	//show(m1, n);
	show_in_file(m1, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head1);
	GE_OMP(m1, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail1);
	cout << endl << endl << endl;
	//show(m1, n);
	sample_output(m1, n);

	cout << "GE_OMP: " << (tail1 - head1) * 1000.0 / freq
		<< "ms" << endl;

	//-----------------------------------------------------------------

	float** m2 = generate(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head2);
	C_GE(m2, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
	cout << "C_GE: " << (tail2 - head2) * 1000.0 / freq
		<< "ms" << endl;
	//show(m2, n);

	//-----------------------------------------------------------------

	m2 = generate(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head2);
	C_GE_OMP(m2, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
	cout << "C_GE_OMP: " << (tail2 - head2) * 1000.0 / freq
		<< "ms" << endl;
	//show(m2, n);

	//-----------------------------------------------------------------

	m2 = generate(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head2);
	C_GE_OMP_Dynamic(m2, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
	cout << "C_GE_OMP_Dynamic: " << (tail2 - head2) * 1000.0 / freq
		<< "ms" << endl;
	//show(m2, n);

	//-----------------------------------------------------------------

	m2 = generate(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head2);
	C_GE_OMP_Static(m2, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
	cout << "C_GE_OMP_Static: " << (tail2 - head2) * 1000.0 / freq
		<< "ms" << endl;
	//show(m2, n);

	//-----------------------------------------------------------------

	m2 = generate(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head2);
	C_GE_OMP_Guided(m2, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
	cout << "C_GE_OMP_Guided: " << (tail2 - head2) * 1000.0 / freq
		<< "ms" << endl;
	//show(m2, n);

	//-----------------------------------------------------------------

	m2 = generate(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head2);
	OMP_Dynamic_Block_GE_ALL(m2, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
	cout << "OMP_Dynamic_Block_GE: " << (tail2 - head2) * 1000.0 / freq
		<< "ms" << endl;
	//show(m2, n);

	//-----------------------------------------------------------------

	m2 = generate(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head2);
	OMP_Dynamic_Rotation_GE_ALL(m2, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
	cout << "OMP_Dynamic_Rotation_GE: " << (tail2 - head2) * 1000.0 / freq
		<< "ms" << endl;
	//show(m2, n);

	//-----------------------------------------------------------------

	m2 = generate(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head2);
	C_GE_OMP_Dynamic_AVX(m2, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
	cout << "C_GE_OMP_Dynamic_AVX: " << (tail2 - head2) * 1000.0 / freq
		<< "ms" << endl;
	//show(m2, n);

	//-----------------------------------------------------------------
	
	m2 = generate(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head2);
	C_GE_OMP_Static_AVX(m2, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
	cout << "C_GE_OMP_Static_AVX: " << (tail2 - head2) * 1000.0 / freq
		<< "ms" << endl;
	//show(m2, n);

	//-----------------------------------------------------------------

	m2 = generate(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head2);
	C_GE_OMP_Guided_AVX(m2, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
	cout << "C_GE_OMP_Guided_AVX: " << (tail2 - head2) * 1000.0 / freq
		<< "ms" << endl;
	//show(m2, n);

	//-----------------------------------------------------------------

	m2 = generate(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head2);
	C_GE_OMP_Dynamic_AUTO_SIMD(m2, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
	cout << "C_GE_OMP_Dynamic_AUTO_SIMD: " << (tail2 - head2) * 1000.0 / freq
		<< "ms" << endl;
	//show(m2, n);

	//-----------------------------------------------------------------


	//float** m4 = generate(n);
	float** m4 = aligned_generate(n);
	//show(m4, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head4);
	AVX_GE(m4, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail4);

	cout << "AVX_GE: " << (tail4 - head4) * 1000.0 / freq
		<< "ms" << endl;
	//show(m4, n);

	//-----------------------------------------------------------------

	global_generate(n);
	//global_reset(n);

	QueryPerformanceCounter((LARGE_INTEGER*)&head5);
	PThread_Dynamic_Block_GE(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail5);

	cout << "PThread_Dynamic_Block_GE: " << (tail5 - head5) * 1000.0 / freq
		<< "ms" << endl;
	//show(mat, n);

	//-----------------------------------------------------------------

	global_generate(n);

	QueryPerformanceCounter((LARGE_INTEGER*)&head6);
	PThread_Dynamic_Rotation_GE(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail6);

	cout << "PThread_Dynamic_Rotation_GE: " << (tail6 - head6) * 1000.0 / freq
		<< "ms" << endl;
	//show(mat, n);

	//-----------------------------------------------------------------

	global_generate(n);

	QueryPerformanceCounter((LARGE_INTEGER*)&head6);
	PThread_Static_OnlyElim_GE(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail6);

	cout << "PThread_Static_OnlyElim_GE: " << (tail6 - head6) * 1000.0 / freq
		<< "ms" << endl;
	//show(mat, n);

	//-----------------------------------------------------------------

	global_generate(n);
	//show(mat, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head6);
	PThread_Static_GE(n);  // �е�����
	QueryPerformanceCounter((LARGE_INTEGER*)&tail6);

	cout << "PThread_Static_GE: " << (tail6 - head6) * 1000.0 / freq
		<< "ms" << endl;
	//show(mat, n);

	//-----------------------------------------------------------------

	global_generate(n);
	//show(mat, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head6);
	PThread_Static_Barrier_GE(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail6);

	cout << "PThread_Static_Barrier_GE: " << (tail6 - head6) * 1000.0 / freq
		<< "ms" << endl;
	//show(mat, n);

	//-----------------------------------------------------------------

	global_generate(n);
	//show(mat, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head6);
	PThread_Static_Barrier_Block_GE(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail6);

	cout << "PThread_Static_Barrier_GE: " << (tail6 - head6) * 1000.0 / freq
		<< "ms" << endl;
	//show(mat, n);

	//-----------------------------------------------------------------

	global_generate(n);
	//show(mat, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head6);
	PThread_Dynamic_Block_GE_AVX(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail6);

	cout << "PThread_Dynamic_Block_GE_AVX: " << (tail6 - head6) * 1000.0 / freq
		<< "ms" << endl;
	//show(mat, n);

	//-----------------------------------------------------------------

	global_generate(n);
	//show(mat, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head6);
	PThread_Dynamic_Rotation_GE_AVX(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail6);

	cout << "PThread_Dynamic_Rotation_GE_AVX: " << (tail6 - head6) * 1000.0 / freq
		<< "ms" << endl;
	//show(mat, n);

	//-----------------------------------------------------------------

	global_generate(n);
	//show(mat, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head6);
	PThread_Static_OnlyElim_GE_AVX(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail6);

	cout << "PThread_Static_OnlyElim_GE_AVX: " << (tail6 - head6) * 1000.0 / freq
		<< "ms" << endl;
	//show(mat, n);

	//-----------------------------------------------------------------

	global_generate(n);
	//show(mat, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head6);
	PThread_Static_GE_AVX(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail6);

	cout << "PThread_Static_GE_AVX: " << (tail6 - head6) * 1000.0 / freq
		<< "ms" << endl;
	//show(mat, n);

	//-----------------------------------------------------------------

	global_generate(n);
	//show(mat, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head6);
	PThread_Static_Barrier_GE_AVX(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail6);

	cout << "PThread_Static_Barrier_GE_AVX: " << (tail6 - head6) * 1000.0 / freq
		<< "ms" << endl;
	//show(mat, n);

	//-----------------------------------------------------------------

	global_generate(n);
	//show(mat, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head6);
	PThread_Static_Barrier_Block_GE_AVX(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail6);

	cout << "PThread_Static_Barrier_GE_AVX: " << (tail6 - head6) * 1000.0 / freq
		<< "ms" << endl;
	//show(mat, n);

	//-----------------------------------------------------------------

	//system("pause");

}