c++ -O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup `python3 -m pybind11 --includes` -I../../ pybsts.cpp -L../.. -lboom -o pybsts`python3-config --extension-suffix`
