#include <unordered_map>
#include <map>
#include <vector>
#include <iostream>
using namespace std;

template <typename T, typename F>
class IndexedHeap
{
public:
    std::map<T, size_t> indices;
    std::map<T, F> priorities;
    std::vector<T> heap;



    bool empty()
    {
        return heap.empty();
    }

    void swap(size_t i, size_t j)
    {
        T temp = heap[i];
        heap[i] = heap[j];
        heap[j] = temp;

        indices[heap[i]] = i;
        indices[heap[j]] = j;
    }

    void insertOrUpdate(T obj, F priority)
    {
        priorities[obj] = priority;
        auto search = indices.find(obj);
        size_t i;
        if (search != indices.end())
        {
            i = search->second;
        }
        else
        {
            heap.push_back(obj);
            i = heap.size()-1;
            indices[obj] = i;
        }
        while (i > 0)
        {
            if (priorities[heap[i]] < priorities[heap[i/2]])
            {
                swap(i, i / 2);
                i = i / 2;
            }
            else
                break;
        }
    }

    T extract()
    {
        T answer = heap[0];
        indices.erase(answer);
        priorities.erase(answer);
        heap[0] = heap.back();
        heap.pop_back();
        indices[heap[0]] = 0;
        size_t i = 0;
        size_t n = heap.size();
        while (2 * i < n)
        {
            if ((2 * i + 1 > n) or (priorities[heap[2 * i]] < priorities[heap[2 * i + 1]]))
            {
                if (priorities[heap[i]] > priorities[heap[2 * i]])
                {
                    swap(i, 2 * i);
                    i = 2 * i;
                }
                else
                    break;
            }
            else
            {
                if (priorities[heap[i]] > priorities[heap[2 * i + 1]])
                {
                    swap(i, 2 * i + 1);
                    i = 2 * i + 1;
                }
                else
                    break;
            }
        }

        return answer;
    }
};