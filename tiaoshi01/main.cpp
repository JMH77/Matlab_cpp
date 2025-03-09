#define _USE_MATH_DEFINES
#include <iostream>
#include <iomanip>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

using namespace std;
using namespace Eigen;

int main() {

    const int N = 64;
    const double fs = 100.0, f1 = 10.0, f2 = 20.0;

    VectorXd time = VectorXd::LinSpaced(N, 0.0, (N-1)/fs);

    VectorXd part1 = 2.0 * sin(2 * M_PI * f1 * time.array());
    VectorXd part2 = 1.0 * sin(2 * M_PI * f2 * time.array());
    VectorXd signal = (part1 + part2).eval();

    FFT<double> fft;
    VectorXcd freqSpectrum;
    fft.fwd(freqSpectrum, signal);

    VectorXd powerSpectrum = freqSpectrum.cwiseAbs2().real() / N;

    cout << "ÆµÂÊ\t\t\t·ù¶È" << endl;
    for(int i=0; i<N/2+1; ++i) {
        cout << setw(12) << left <<(i * fs / N) << "Hz\t" <<
        setw(12) << right << powerSpectrum(i) << endl;
    }

        return 0;
}




// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.