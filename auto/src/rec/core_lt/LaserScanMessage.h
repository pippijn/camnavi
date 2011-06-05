#ifndef _REC_CORELT_LaserScanMessage_H_
#define _REC_CORELT_LaserScanMessage_H_

#include <rec/core_lt/Vector.h>
#include <boost/cstdint.hpp>

#include <vector>
#include <string>
#include <cstring>

namespace rec
{
	namespace core_lt
	{
		class LaserScanMessage
		{
		public:
			LaserScanMessage();

			LaserScanMessage( const char* data, const int dataSize );

			virtual ~LaserScanMessage();

			static const int minimumEncodedDataSize = 42;

			/**
			Returns the size of the buffer given to the encode function.
			*/
			virtual int encodedDataSize() const;

			/**
			Encode this data to buffer. The size of buffer must be at least encodedDataSize.
			@return num bytes encoded
			@see encodedDataSize;
			*/
			virtual int encode( char* buffer ) const;

			virtual bool operator== ( const LaserScanMessage& other ) const;

			virtual bool operator!= ( const LaserScanMessage& other ) const;

			bool isEmpty() const;

			void clear();

			class Header
			{
			public:
				Header()
					: seq( 0 )
					, stamp( 0 )
				{
				}

				Header( const char* data, const int dataSize )
					: seq( 0 )
					, stamp( 0 )
				{
					decode( data, dataSize );
				}

				static const int minimumEncodedDataSize = 10;

				int encodedDataSize() const
				{
					return
						  4 // seq
						+ 4 // stamp
						+ 1 // frame_id size
						+ frame_id.size();
				}

				/**
				@return num bytes encoded
				*/
				int encode( char* buffer ) const
				{
					boost::uint32_t* uint32p = reinterpret_cast<boost::uint32_t*>( buffer );
					(*uint32p++) = seq;
					(*uint32p++) = stamp;
					buffer += 8;

					unsigned char* up = reinterpret_cast<unsigned char*>( buffer );
					*up = static_cast<unsigned char>( frame_id.size() );
					++buffer;

					memcpy( buffer, frame_id.data(), frame_id.size() );
					buffer += frame_id.size();

					return encodedDataSize();
				}

				/**
				@return num bytes decoded
				*/
				int decode( const char* data, const int dataSize )
				{
					if( dataSize >= minimumEncodedDataSize )
					{
						const boost::uint32_t* uint32p = reinterpret_cast<const boost::uint32_t*>( data );

						seq = *(uint32p++);
						stamp = *uint32p;
						data += 8;

						const unsigned char frame_id_size = *( reinterpret_cast<const unsigned char*>( data ) );
						++data;

						frame_id = std::string( data, frame_id_size );
						data += frame_id_size;

						return encodedDataSize();
					}
					else
					{
						return 0;
					}
				}

				void clear()
				{
					seq = 0;
					stamp = 0;
					frame_id.clear();
				}

				bool operator== ( const Header& other ) const
				{
					return !operator!=( other );
				}

				bool operator!= ( const Header& other ) const
				{
					if( seq != other.seq
						|| stamp != other.stamp
						|| frame_id != other.frame_id )
					{
						return true;
					}
					return false;
				}

				boost::uint32_t seq;
				boost::uint32_t stamp;
				std::string frame_id;
			} header;

			float angle_min;
			float angle_max;
			float angle_increment;
			float time_increment;
			float scan_time;
			float range_min;
			float range_max;

			/**
			Measurements in meters
			*/
			rec::core_lt::Vector< float > ranges;

			rec::core_lt::Vector< float > intensities;
		};

		inline LaserScanMessage::LaserScanMessage()
		{
			clear();
		}

		inline LaserScanMessage::LaserScanMessage( const char* data, const int dataSize )
		{
			if( dataSize >= minimumEncodedDataSize )
			{
				data += header.decode( data, dataSize );

				const float* floatp = reinterpret_cast<const float*>( data );

				angle_min = *( floatp++ );
				angle_max = *( floatp++ );
				angle_increment = *( floatp++ );
				time_increment = *( floatp++ );
				scan_time = *( floatp++ );
				range_min = *( floatp++ );
				range_max = *( floatp++ );

				data += 28;

				const boost::uint16_t* uint16p = reinterpret_cast<const boost::uint16_t*>( data );
				const boost::uint16_t ranges_size = *(uint16p++);
				const boost::uint16_t intensities_size = *(uint16p++);

				data += 4;

				floatp = reinterpret_cast<const float*>( data );
				ranges.resize( ranges_size );
				for( unsigned int i=0; i<ranges_size; ++i )
				{
					ranges[i] = *(floatp++);
				}

				intensities.resize( intensities_size );
				for( unsigned int i=0; i<intensities_size; ++i )
				{
					intensities[i] = *(floatp++);
				}

				intensities.resize( ranges.size(), 0 );
			}
			else
			{
				clear();
			}
		}

		inline LaserScanMessage::~LaserScanMessage()
		{
		}

		inline int LaserScanMessage::encodedDataSize() const
		{
			int length =
				header.encodedDataSize() //header
				+ 4 //angle_min
				+ 4 //angle_max
				+ 4 //angle_increment
				+ 4 //time_increment
				+ 4 //scan_time
				+ 4 //range_min
				+ 4 //range_max
				+ 2 //ranges.size()
				+ 2 //intensities.size()
				+ 4 * ranges.size() //ranges
				+ 4 * intensities.size() //intensities
				;

			return length;
		}

		inline int LaserScanMessage::encode( char* buffer ) const
		{
			buffer += header.encode( buffer );
			
			float* floatp = reinterpret_cast<float*>( buffer );
			*(floatp++) = angle_min;
			*(floatp++) = angle_max;
			*(floatp++) = angle_increment;
			*(floatp++) = time_increment;
			*(floatp++) = scan_time;
			*(floatp++) = range_min;
			*(floatp++) = range_max;

			buffer += 28;

			boost::uint16_t* uint16p = reinterpret_cast<boost::uint16_t*>( buffer );
			*(uint16p++) = static_cast<boost::uint16_t>( ranges.size() );
			*(uint16p++) = static_cast<boost::uint16_t>( intensities.size() );

			buffer += 4;

			floatp = reinterpret_cast<float*>( buffer );

			for( int i=0; i<ranges.size(); ++i )
			{
				*(floatp++) = ranges[ i ];
			}
			buffer += 4 * ranges.size();

			for( int i=0; i<intensities.size(); ++i )
			{
				*(floatp++) = intensities[ i ];
			}
			buffer += 4 * intensities.size();

			return encodedDataSize();
		}

		inline bool LaserScanMessage::operator== ( const LaserScanMessage& other ) const
		{
			return !operator!=( other );
		}

		inline bool LaserScanMessage::operator!= ( const LaserScanMessage& other ) const
		{
			if( header != other.header
				|| angle_min != other.angle_min
				|| angle_max != other.angle_max
				|| angle_increment != other.angle_increment
				|| time_increment != other.time_increment
				|| scan_time != other.scan_time
				|| range_min != other.range_min
				|| range_max != other.range_max
				|| ranges != other.ranges
				|| intensities != other.intensities )
			{
				return true;
			}
			return false;
		}

		inline bool LaserScanMessage::isEmpty() const
		{
			return ranges.isEmpty();
		}

		inline void LaserScanMessage::clear()
		{
			header.clear();

			angle_min = 0.0f;
			angle_max = 0.0f;
			angle_increment = 0.0f;
			time_increment = 0.0f;
			scan_time = 0.0f;
			range_min = 0.0f;
			range_max = 0.0f;

			ranges.clear();
			intensities.clear();
		}
	}
}

#ifdef HAVE_QT
#include <QMetaType>
Q_DECLARE_METATYPE(rec::core_lt::LaserScanMessage)
#endif

#endif //_REC_CORELT_LaserScanMessage_H_
