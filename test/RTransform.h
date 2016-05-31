#include <vector>
#include <iostream>

class RTransformer {
public:
	RTransformer();
	~RTransformer();
	std::vector <double> RTransform(std::vector< std::vector<unsigned char> > im, int cols, int rows, int N);
};

