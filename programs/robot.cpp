#include "robot.h"

int
main (int argc, char *argv[])
{
  std::string hostname = "172.26.1.1";
  //std::string hostname = "172.26.1.2";

  if (argc > 1)
    hostname = argv[1];

  Robot robot (hostname);
  robot.run ();

  return 0;
}
