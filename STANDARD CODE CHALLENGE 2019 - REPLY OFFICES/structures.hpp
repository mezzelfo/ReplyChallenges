#ifndef STRUCTURES_HPP
#define STRUCTURES_HPP

#include <fstream>
#include <array>
#include <stdexcept>

class Matrix
{
    int** ptr;
    public:
    const int N, M;
    Matrix(int n, int m) : N(n), M(m)
    {
        ptr = new int*[N];
        for (int i = 0; i < N; i++) ptr[i] = new int[M];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < M; j++)
                at(i,j) = 0;
    }
    Matrix(const Matrix& mat) :  N(mat.N), M(mat.M)
    {
        ptr = new int*[N];
        for (int i = 0; i < N; i++) ptr[i] = new int[M];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < M; j++)
                at(i,j) = mat.at(i,j);
    }
    ~Matrix()
    {
        for (int i = 0; i < N; ++i) delete [] ptr[i];
        delete [] ptr;
    }
    int& at(const int i, const int j)
    {
        if ((i < 0) or (i >= N) or (j < 0) or (j >= M))
            throw std::runtime_error("Indici non corretti");
        return ptr[i][j];
    }
    const int& at(const int i, const int j) const
    {
        if ((i < 0) or (i >= N) or (j < 0) or (j >= M))
            throw std::runtime_error("Indici non corretti");
        return ptr[i][j];
    }
};

struct CustomerHQ
{
    int x,y,reward;
    //Matrix positiveZone;
};

std::istream& operator >> (std::istream& in, CustomerHQ& c) { return (in >> c.x >> c.y >> c.reward); }
bool operator < (const CustomerHQ& a, const CustomerHQ& b) { return (a.reward > b.reward); }



#endif

