#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

int main(int iArgC, char** iArgV) {
  //int i;
  //cout<<iArgC<<endl;
  if (iArgC > 1) {
    string is_save_nis = iArgV[1];
    cout<<is_save_nis<<endl;
		if (is_save_nis.compare("save-nis") == 0) {
			ofstream out_file;
		  cout<<iArgV[2]<<endl;
		  out_file.open(iArgV[2], ofstream::out);
		  if (out_file.is_open()) {
		    out_file<<"Col1"<<setw(10);
		    out_file<<"Col2"<<setw(10);
		    out_file<<"Col3"<<setw(10);
		    out_file<<"Col4"<<setw(10);
		    out_file<<"Col5"<<setw(10);
		    out_file.close();
		  }
		}
    /*for (i = 0; i < iArgC; i++) {
      cout<<iArgV[i]<<endl;
    }*/
  }
  return 0;
}
