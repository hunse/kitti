
/* a 3d volume class */

#ifndef VOLUME_H_2014_08_15
#define VOLUME_H_2014_08_15

template <class T>
class volume {
public:
    /* create an volume */
    volume(const int width, const int height, const int depth, const bool init = true);

    /* delete a volume */
    ~volume();

    /* get the width of a volume. */
    int width() const { return w; }

    /* get the height of a volume. */
    int height() const { return h; }

    /* get the depth of a volume. */
    int depth() const { return d; }

    /* volume data. */
    T *data;

    /* row pointers. */
    T ***access;

    inline T* operator ()(const int x, const int y) { return access[y][x]; }
    inline T& operator ()(const int x, const int y, const int z) { return access[y][x][z]; }

    /* inline T* operator ()(const int x, const int y) { return &data[y * w * d + x * d]; } */
    /* inline T& operator ()(const int x, const int y, const int z) { return data[y * w * d + x * d + z]; } */

private:
    int w, h, d;
};


template <class T>
volume<T>::volume(const int width, const int height, const int depth, const bool init) {
    w = width;
    h = height;
    d = depth;
    data = new T[w * h * d];  // allocate space for volume data
    access = new T**[h];   // allocate space for row pointers

    // initialize row pointers
    T* dataij = data;
    for (int i = 0; i < h; i++) {
        access[i] = new T*[w];
        for (int j = 0; j < w; j++, dataij += d)
            access[i][j] = dataij;
    }

    if (init)
        memset(data, 0, w * h * d * sizeof(T));
}

template <class T>
volume<T>::~volume() {
    delete [] data;
    for (int i = 0; i < h; i++)
        delete [] access[i];
    delete [] access;
}

#endif
