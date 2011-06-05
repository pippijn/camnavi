package rec.robotino.com;

/**
 * This class is responsible for loading the Robotino native libraries.
 */
final class RobotinoLibraryLoader
{
    private static volatile boolean loaded = false;

    public static synchronized void loadLibraries()
    {
        if (!loaded)
        {
            String robDir = System.getenv("OPENROBOTINOAPI_DIR");

            String os;
            if (System.getProperty("os.name").toLowerCase().startsWith("win"))
            {
                os = "/bin/win32/";
                System.load(robDir + "/1" + os + System.mapLibraryName("QtCore4"));
				System.load(robDir + "/1" + os + System.mapLibraryName("QtNetwork4"));
            }
            else
            {
                os = "/lib/linux/";
                System.load( "/usr/lib/libQtCore.so.4");
				System.load( "/usr/lib/libQtNetwork.so.4");
            }

            System.load(robDir + "/1" + os + System.mapLibraryName("rec_core_lt"));
            System.load(robDir + "/1" + os + System.mapLibraryName("rec_robotino_com"));
            System.load(robDir + "/1" + os + System.mapLibraryName("rec_robotino_com_java"));

            loaded = true;
        }
    }
}
