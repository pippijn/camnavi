//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

//Untested

#include <sys/ipc.h>
#include <sys/types.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <errno.h>

using rec::core_lt::Lock;

class SharedMemoryImpl
{
public:
  int _shmid;
  int _semid;
  sembuf _sb;
};

template< typename sharedType >
SharedMemory< sharedType >::SharedMemory( int key )
{
  impl = new SharedMemoryImpl();
  this->key = key;
  data = NULL;

  // try to open an existing shared memory segment
  int shmflag = 0666;
  int size = sizeof( sharedType );
  impl->_shmid = shmget( key, size, shmflag );
  if( impl->_shmid == -1 )
  {
    //try to create a new shared memory segment
    impl->_shmid = shmget( key, size, shmflag | IPC_CREAT);
    if( impl->_shmid == -1 )
    {
      std::ostringstream os;
      os << "shmget Error. errno = " << errno;
      throw rec::core_lt::Exception(os.str());
    }
    owner = true;
  }
  else
    owner = false;

  data = (sharedType*) shmat( impl->_shmid, NULL, 0 );
  if( data == (sharedType*) -1 )
    throw rec::core_lt::Exception("error 2");
  
  // initialize semaphore
  impl->_semid = semget( key + 1, 1, shmflag | IPC_CREAT );
  if( impl->_semid == -1 )
  {
    std::ostringstream os;
    os << "semget failed. errno = " << errno;
    throw rec::core_lt::Exception(os.str());
  }

  //TODO check this
  //if(owner)
  {
    if( semctl( impl->_semid, 0, SETVAL, 1 ) == -1 )
      throw rec::core_lt::Exception("semctl failed");
  }

  impl->_sb.sem_num = 0;
  impl->_sb.sem_flg = SEM_UNDO;

  if( owner )
  {    
    // in-place new to initialize share-memory
    new( data ) sharedType;
  }

  if( owner )
    std::cout << "Shared memory has been successfully created!" << std::endl;
  else
    std::cout << "Shared memory has been successfully opened!" << std::endl;
}

template< typename sharedType >
SharedMemory< sharedType >::~SharedMemory()
{
  shmdt( (const void*) data );
  if( owner )
  {
    // destroy object that was created with in-place new
//    data->~sharedType();
    shmid_ds s;
    shmctl( impl->_shmid, IPC_RMID, &s );
  }

  delete impl;

  std::cout << "Shared memory has been successfully destroyed!" << std::endl;
}

template< typename sharedType >
void SharedMemory< sharedType >::lock()
{
  impl->_sb.sem_op = -1;
  semop( impl->_semid, &impl->_sb, 1 );
  if( semctl( impl->_semid, 0, GETVAL ) != 0 )
  {
    throw rec::core_lt::Exception( "lock failed" );
  }
}

template< typename sharedType >
void SharedMemory< sharedType >::unlock()
{
  impl->_sb.sem_num = 0;
  impl->_sb.sem_flg = 0;
  impl->_sb.sem_op = 1;
  semop( impl->_semid, &impl->_sb, 1 );
  if( semctl( impl->_semid, 0, GETVAL ) != 0 )
  {
    throw rec::core_lt::Exception( "unlock failed" );
  }
}

template< typename sharedType >
sharedType* SharedMemory< sharedType >::getData() const
{
  return data;
}
