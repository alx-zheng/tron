#include <bits/stdc++.h>
#include <iostream>

extern "C" {
  Simulator* New_Simulator() {
    return new Simulator();
  }
  char Sim_simulator(Simulator* sim) {
    return sim->simulate();
  }
}

class Simulator {
  public:
    float heuristic(std::pair<int, int> location, std::vector<vector<int>> state){
      return -1;
    }

    char simulate() {
      return 'U';
    }
}
