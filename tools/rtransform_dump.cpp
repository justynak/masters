// Standalone harness around the thesis R-transform (test/RTransform.cpp).
// Used to capture golden reference values for the refactoring test suite,
// with no Python/Cython dependency.
//
// Input file format (text):
//   line 1: "<dim0> <dim1>"  -- matrix dimensions
//   then dim0 lines of dim1 space-separated pixel values (0..255)
//
// The matrix is passed to RTransform() exactly the way the Python pipeline
// does: numpy shape[0] as the "cols" parameter, shape[1] as "rows"
// (see imageshow.py: `x, y = sil.shape; rT.rTransform(sil, x, y, 64)`).
//
// Usage: rtransform_dump <input.txt> [N]     (N defaults to 64)
// Output: N lines, one normalised R-transform value per angle, "%.17g".

#include "../test/RTransform.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <input.txt> [N]\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 2;
    }
    int N = (argc > 2) ? std::atoi(argv[2]) : 64;

    int dim0 = 0, dim1 = 0;
    in >> dim0 >> dim1;
    if (dim0 <= 0 || dim1 <= 0) {
        std::cerr << "bad dimensions\n";
        return 2;
    }

    std::vector<std::vector<unsigned char> > im(dim0, std::vector<unsigned char>(dim1));
    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            int v;
            if (!(in >> v)) {
                std::cerr << "truncated input at " << i << "," << j << "\n";
                return 2;
            }
            im[i][j] = (unsigned char)v;
        }
    }

    RTransformer rt;
    std::vector<double> result = rt.RTransform(im, dim0, dim1, N);

    for (size_t i = 0; i < result.size(); ++i)
        std::printf("%.17g\n", result[i]);
    return 0;
}
