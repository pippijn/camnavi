#include <openvidia/openvidia32.h>

/*
class AutoTimer 
{
    private:
        Timer t;
        Recorder r;
    public:
        AutoTimer();
        void start();
        void end();
};
*/

AutoTimer::AutoTimer() 
{
    // begin in a state sampling times
    samplingState = true;
    sleepTime = 0;
}

void AutoTimer::start()
{
    if( samplingState ) {
        t.start();
    }
}

void AutoTimer::end()
{
    if( samplingState ) {
        glFinish();
        t.end();
        r.addSample( t.elapsedTime() );
    }
    
    //after enough samples, record mean sleep time.
    if( r.numSamples() == 120 && samplingState ) {
        samplingState = false;
        sleepTime = r.mean();
        tv.tv_sec = 0; tv.tv_usec = sleepTime;
        cerr<<"Free time found : "<<tv.tv_usec <<" micro seconds" <<endl;
    }

    // if done sampling, just sleep
    if( !samplingState ) {
        tv.tv_sec = 0; tv.tv_usec = sleepTime;
        select(0,0,0,0, &tv);
    }

}
