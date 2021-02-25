#ifndef COMM_BUF_
#define COMM_BUF_

#include <iostream>
#include <map>
#include <mutex>

class comm_buffer {
private:
    std::map <int, int> comm_buf;
    std::mutex mtx;
public:
    comm_buffer(int __len = 5000) {
        for (int i = 0; i < __len; i++) {
            comm_buf[i] = 0;
        }
    }

    void notify_with_tag(int __tag) {
        mtx.lock();
        comm_buf[__tag] = 1;
        mtx.unlock();
    }

    void wait_with_tag(int __tag) {
        while (true) {
            if (comm_buf[__tag] == 1) break;
        }
    }
};

#endif