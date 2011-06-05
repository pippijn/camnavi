#ifndef _REC_ROBOTINO_COM_V1_MEANBUFFER_H_
#define _REC_ROBOTINO_COM_V1_MEANBUFFER_H_

namespace rec
{
	namespace robotino
	{
		namespace com
		{
			namespace v1
			{
				class MeanBuffer
				{
				public:
					MeanBuffer( unsigned int size = 50, float startValue = 0.0f )
						: _size( size )
						, _buffer( new float[ _size ] )
						, _isEmpty( false )
					{
						reset( startValue );
					}

					~MeanBuffer()
					{
						delete [] _buffer;
					}

					void reset( float startValue = 0.0f )
					{
						if( _isEmpty )
						{
							return;
						}

						_currentIndex = 0;
						_bufferSum = 0.0f;
						_bufferMean = startValue;

						for( unsigned int i=0; i<_size; i++ )
						{
							_buffer[i] = startValue;
						}
						_bufferSum = startValue * _size;
						
						_isEmpty = true;
					}

					void add( float value )
					{
						_currentIndex = ( _currentIndex + 1 ) % _size;
						_bufferSum -= _buffer[ _currentIndex ];
						_buffer[ _currentIndex ] = value;
						_bufferSum += value;
						_bufferMean = _bufferSum / _size;
						_isEmpty = false;
					}

					float mean() const
					{
						return _bufferMean;
					}

					bool isEmpty() const { return _isEmpty; }

				private:
					const unsigned int _size;
					float* _buffer;
					unsigned int _currentIndex;
					float _bufferSum;
					float _bufferMean;
					bool _isEmpty;
				};
			};
		}
	}
}

#endif //_REC_ROBOTINO_COM_V1_MEANBUFFER_H_
