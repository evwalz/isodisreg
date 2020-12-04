#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <pybind11/stl.h>
#include <vector>
#include <numeric>


/*
* reshape output to matrix *no need*
* try to not use memcpy for input *no need*
* test if using np.contiguous imrpoves performance *no need*
* eigen has a sum function which can be applied directly to numpy array input
* parallelize C++ Code
*/
namespace py = pybind11;

//py::array_t<double>
py::array_t<double> isocdf_seq(py::array_t<double>& w, py::array_t<double>& W, py::array_t<double>& Y, py::array_t<int>& posY, py::array_t<double>& y) {
    // Input
    std::vector<double> array_y(y.size());
    std::vector<double> array_w(w.size());

    // copy py::array -> std::vector
    std::memcpy(array_y.data(), y.data(), y.size() * sizeof(double));
    std::memcpy(array_w.data(), w.data(), w.size() * sizeof(double));

    double inf = std::numeric_limits<double>::infinity();

    //read array buffer input
    //py::buffer_info bufw = w.request(), bufW = W.request();
    py::buffer_info bufY = Y.request(), bufposY = posY.request(), bufW = W.request();

    //double* ptrw = (double*)bufw.ptr;
    double* ptrY = (double*)bufY.ptr;
    double* ptrW = (double*)bufW.ptr;
    int* ptrposY = (int*)bufposY.ptr;
    //double* ptry = (double*)bufy.ptr;



    py::ssize_t m = array_w.size();
    py::ssize_t mY = array_y.size();

    //int indxyMax = static_cast<int>(mY - 1);
    double yMax = array_y[mY-1];
   
    //double yMax = 1;
    // K is the maximal index such that Y[K] < Y[K + 1]
    py::ssize_t K = bufY.shape[0] - 1;
    while (ptrY[K] == ptrY[K - 1]) K--;
    K--;
    if (ptrY[K] < yMax) yMax = ptrY[K];

    // yInd is used to check when to store the CDF at y[yInd]
    int yInd = 0;
    array_y.push_back(inf);
    /*ptry.push_back(inf);*/
    while (array_y[yInd] < ptrY[0]) yInd++;

    // Prepare object containers
    std::vector<double> z0(m);
    std::vector<int> PP(m + 1);
    std::vector<double> WW(m + 1);
    std::vector<double> MM(m + 1);
    
    // Prepare output
    double *CDF = new double [m*mY];
    for (int i = 0; i < m * mY; i++) CDF[i] = 1;
    //std::vector<double>* CDF = new std::vector<double>[m*mY];

    //CDF.reserve(m * mY);
    /*
    py::array_t<double> cdf = py::array_t<double>(m*mY);
    py::buffer_info bufcdf = cdf.request();
    double* ptrcdf = (double*)bufcdf.ptr;*/
    PP[0] = -1;
    for (int i = 0; i < m * yInd; i++) CDF[i] = 0;
    int indxcdf = m * yInd;
    // First iteration
    int d = 1;
    int j0 = ptrposY[0]; // -1 FOR R INDEXING
    z0[j0] = ptrW[0] / array_w[j0];
    PP[1] = j0;
    //WW[1] = sum(ptrw, 0, j0 + 1);
    WW[1] = accumulate(array_w.begin(), array_w.begin() + j0 + 1, 0);
    MM[0] = inf;
    MM[1] = ptrW[0] / WW[1];
    if (j0 < m - 1) {
        d++;
        PP[2] = m - 1;
        //WW[2] = sum(ptrw, j0, m);
        WW[2] = accumulate(array_w.begin() + j0, array_w.begin() + m, 0);
    }
    int upper;
    int countloop;
    // Store results if needed; keep track of how many results have been stored
    if ((ptrY[0] < ptrY[1]) && (array_y[yInd] < ptrY[1])) {
        for (int l = 1; l <= d; l++) {
            int len = PP[l] - PP[l - 1];
            upper = indxcdf + len;
            double val = MM[l];
            for (int i = indxcdf; i < upper; i++) CDF[i] = val;
            indxcdf = upper;
        }
        yInd++;
        countloop = 1;
        while (array_y[yInd] < ptrY[1]) {
            upper = indxcdf + m;
            //startindx = len + indxcdf + countindx;
            for (int j = indxcdf; j < upper; j++) CDF[j] = CDF[j - m*countloop];
            //CDF.insert(CDF.end(), CDF.end() - m, CDF.end());
            indxcdf = upper;
            countloop++;
            yInd++;
        }
    }

    // Prepare objects for loop
    int k = 1;
    int b0;
    int a0;
    int s0;
    int dz;
    std::vector<int> remPP;
    std::vector<double> remWW;
    std::vector<double> remMM;
    
  
    while (ptrY[k] <= yMax) {
        // Update z0
        j0 = ptrposY[k]; // -1 FOR R INDEXING
        z0[j0] = z0[j0] + ptrW[k] / array_w[j0];

        // Find partition to which j0 belongs
        s0 = 0;
        while (j0 > PP[s0]) s0++;
        a0 = PP[s0 - 1] + 1;
        b0 = PP[s0];
        dz = d;

        // Copy tail of vector
        if (b0 < m - 1) {
            //remPP = PP[slice(s0 + 1, dz - s0 + 1, 1)];
            //remMM = MM[slice(s0 + 1, dz - s0 + 1, 1)];
            //remWW = WW[slice(s0 + 1, dz - s0 + 1, 1)];
            remPP.assign(PP.begin() + s0 + 1, PP.begin() + dz + 1);
            remMM.assign(MM.begin() + s0 + 1, MM.begin() + dz + 1);
            remWW.assign(WW.begin() + s0 + 1, WW.begin() + dz + 1);
        }

        // Update value on new partition
        d = s0;
        PP[s0] = j0;
        //WW[s0] = sum(ptrw, a0, j0 + 1);
        WW[s0] = accumulate(array_w.begin() + a0, array_w.begin() + j0 + 1, 0);
        MM[s0] = inner_product(array_w.begin() + a0, array_w.begin() + j0 + 1, z0.begin() + a0, 0.0) / WW[s0];
        //MM[s0] = innerproduct(ptrw, z0, a0, j0 + 1) / WW[s0];

        // Pooling
        while (MM[d - 1] <= MM[d]) {
            d--;
            MM[d] = WW[d] * MM[d] + WW[d + 1] * MM[d + 1];
            WW[d] = WW[d] + WW[d + 1];
            MM[d] = MM[d] / WW[d];
            PP[d] = PP[d + 1];
        }

        // Add new partitions, pool
        if (j0 < b0) {
            for (int i = j0 + 1; i <= b0; i++) {
                d++;
                PP[d] = i;
                WW[d] = array_w[i];
                MM[d] = z0[i];
                while (MM[d - 1] <= MM[d]) {
                    d--;
                    MM[d] = WW[d] * MM[d] + WW[d + 1] * MM[d + 1];
                    WW[d] = WW[d] + WW[d + 1];
                    MM[d] = MM[d] / WW[d];
                    PP[d] = PP[d + 1];
                }
            }
        }

        // Copy (if necessary)
        if (b0 < m - 1) {
            int l0 = dz - s0;
            for (int i = 0; i < l0; i++) {
                int pos = i + d + 1;
                PP[pos] = remPP[i];
                MM[pos] = remMM[i];
                WW[pos] = remWW[i];
            }
            d = d + l0;
        }

        // increase k
        k++;

        // Store in matrix (if necessary)
        if (ptrY[k - 1] < ptrY[k]) {
            while (array_y[yInd] < ptrY[k - 1]) yInd++;
            if (array_y[yInd] < ptrY[k]) {
                for (int l = 1; l <= d; l++) {
                    int len = PP[l] - PP[l - 1];
                    double val = MM[l];
                    upper = indxcdf + len;
                    for (int i = indxcdf; i < upper; i++) CDF[i] = val;
                    indxcdf = upper;
                }
                yInd++;
                countloop = 1;
                while (array_y[yInd] < ptrY[k]) {
                    upper = indxcdf + m;
                    for (int j = indxcdf; j < upper; j++) CDF[j] = CDF[j - m*countloop];
                    //CDF.insert(CDF.end(), CDF.end() - m, CDF.end());
                    yInd++;
                    countloop++;
                    indxcdf = upper;
                }
            }
        }
    }
    
    // Transform vector 'CDF' to matrix
    //if (array_y[mY - 1] >= ptrY[bufY.shape[0] - 1]) {
        //int len = m * mY - CDF.size();
        //for (int i = 0; i < len; i++) CDF.push_back(1.0);
    //}

    // CDF should be converted to a matrix with length(unique(y)) (mY)
    // columns (here, it is a vector where the columns are stacked).
    //py::array cdfout = py::cast(CDF);
    py::capsule free_when_done(CDF, [](void* f) {
        //auto foo = reinterpret_cast<std::vector<double>*>(f);
        double * foo = reinterpret_cast<double *>(f);
        delete foo;
        });
    return py::array_t<double>({ m*mY }, // shape
        { sizeof(double) },      // stride
        //CDF->data(),   // data pointer
        CDF,
        free_when_done);
}



PYBIND11_MODULE(_isodisreg, m) {
    m.def("isocdf_seq", &isocdf_seq);
}
/*
<%
setup_pybind11(cfg)
%>
*/







