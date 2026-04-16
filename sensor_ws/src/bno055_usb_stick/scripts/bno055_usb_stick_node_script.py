#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from bno055_usb_stick_py import BnoUsbStick
import numpy as np
import threading
from time import sleep

class BNO055USBSTICKNode(Node):
    def __init__(self):
        super().__init__('bno055_usb_stick_node')
        
        # Initialize shared data structures FIRST
        self.imu_data = None  
        self.data_lock = threading.Lock()
        self.is_initialized = False
        
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.publisher = self.create_publisher(Imu, 'imu/data', qos_profile)

        # Initialize BNO055 USB Stick
        try:
            self.bno_usb_stick = BnoUsbStick(port='/dev/ttyACM0')
            self.get_logger().info("BNO055 USB Stick connected on /dev/ttyACM0")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to BNO055: {e}")
            raise
        
        # Give device time to initialize and clear buffers thoroughly
        self.get_logger().info("Waiting for device to stabilize...")
        sleep(1.5)  # Increased wait time
        
        # Clear serial buffers multiple times
        try:
            if hasattr(self.bno_usb_stick, 'ser'):
                for i in range(3):
                    self.bno_usb_stick.ser.reset_input_buffer()
                    self.bno_usb_stick.ser.reset_output_buffer()
                    sleep(0.1)
                # Read and discard any remaining data
                if self.bno_usb_stick.ser.in_waiting > 0:
                    discarded = self.bno_usb_stick.ser.read(self.bno_usb_stick.ser.in_waiting)
                    self.get_logger().info(f"Discarded {len(discarded)} bytes from buffer")
                self.get_logger().info("Serial buffers cleared")
        except Exception as e:
            self.get_logger().warn(f"Could not clear buffers: {e}")
        
        sleep(0.5)
        
        # Reset and configure the IMU with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.get_logger().info(f"IMU reset attempt {attempt + 1}/{max_retries}")
                self.imu_reset()
                self.is_initialized = True
                break
            except Exception as e:
                self.get_logger().error(f"Reset attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    self.get_logger().info("Retrying in 2 seconds...")
                    sleep(2)
                    # Clear buffers again before retry
                    if hasattr(self.bno_usb_stick, 'ser'):
                        self.bno_usb_stick.ser.reset_input_buffer()
                        self.bno_usb_stick.ser.reset_output_buffer()
                else:
                    self.get_logger().error("All reset attempts failed")
                    self.get_logger().error("Troubleshooting steps:")
                    self.get_logger().error("1. Check if you're in dialout group: groups | grep dialout")
                    self.get_logger().error("2. Check device: ls -l /dev/ttyACM0")
                    self.get_logger().error("3. Check if device is busy: sudo lsof /dev/ttyACM0")
                    self.get_logger().error("4. Try unplugging and replugging the device")
                    self.get_logger().error("5. Try a different USB port")
                    raise
        
        # Start the data streaming thread
        self.reading_thread = threading.Thread(target=self.imu_streaming)
        self.reading_thread.daemon = True
        self.reading_thread.start()
        
        # Create timer AFTER initialization is complete
        self.timer = self.create_timer(1/100, self.timer_callback)
        
        self.get_logger().info("BNO055 USB Stick initialized successfully.")

    def safe_read_register(self, addr, retries=3):
        """Read register with retry logic"""
        for attempt in range(retries):
            try:
                value = self.bno_usb_stick.read_register(addr)
                return value
            except Exception as e:
                if attempt < retries - 1:
                    self.get_logger().warn(f"Read register 0x{addr:02X} failed (attempt {attempt + 1}), retrying: {e}")
                    sleep(0.1)
                else:
                    raise

    def safe_write_register(self, addr, value, retries=3):
        """Write register with retry logic"""
        for attempt in range(retries):
            try:
                self.bno_usb_stick.write_register(addr, value)
                return
            except Exception as e:
                if attempt < retries - 1:
                    self.get_logger().warn(f"Write register 0x{addr:02X} failed (attempt {attempt + 1}), retrying: {e}")
                    sleep(0.1)
                else:
                    raise

    def imu_reset(self):
        """Reset and configure the BNO055 IMU"""
        # Register addresses
        opr_mode_addr = 0x3D
        sys_trigger_addr = 0x3F
        unit_sel_addr = 0x3B
        
        # Read initial mode
        self.get_logger().info("Reading initial operation mode...")
        mode = self.safe_read_register(opr_mode_addr)
        self.get_logger().info(f"Initial operation mode: 0x{mode:02X} ({mode:08b})")
        
        # Software reset
        self.get_logger().info("Performing software reset...")
        reset_val = 1 << 5  # 0x20
        self.safe_write_register(sys_trigger_addr, reset_val)
        sleep(1.0)
        
        # Clear buffers after reset
        if hasattr(self.bno_usb_stick, 'ser'):
            self.bno_usb_stick.ser.reset_input_buffer()
            self.bno_usb_stick.ser.reset_output_buffer()
        sleep(0.5)
        
        # Set to CONFIG mode
        self.get_logger().info("Setting CONFIG mode...")
        config_mode = 0b00000000
        self.safe_write_register(opr_mode_addr, config_mode)
        sleep(0.2)
        mode = self.safe_read_register(opr_mode_addr)
        self.get_logger().info(f"Config mode: 0x{mode:02X} ({mode:08b})")
        
        # Set units (m/s² for accel, rad/s for gyro, Celsius)
        unit_sel = self.safe_read_register(unit_sel_addr)
        self.get_logger().info(f"Initial unit selection: 0x{unit_sel:02X} ({unit_sel:08b})")
        
        si_units = 0b00000110  # Android orientation, m/s², rad/s, Celsius, degrees
        self.safe_write_register(unit_sel_addr, si_units)
        sleep(0.1)
        
        # Set calibration parameters
        self.get_logger().info("Setting calibration parameters...")
        calibration_data = [
            (0x55, 0xFA), (0x56, 0xFF),  # acc_offset_x
            (0x57, 0xFD), (0x58, 0xFF),  # acc_offset_y
            (0x59, 0xDF), (0x5A, 0xFF),  # acc_offset_z
            (0x61, 0xFF), (0x62, 0xFF),  # gyr_offset_x
            (0x63, 0x03), (0x64, 0x00),  # gyr_offset_y
            (0x65, 0x00), (0x66, 0x00),  # gyr_offset_z
            (0x67, 0xE8), (0x68, 0x03),  # acc_radius
            (0x69, 0xE0), (0x6A, 0x01),  # mag_radius
        ]
        
        for addr, value in calibration_data:
            self.safe_write_register(addr, value)
        
        sleep(0.2)
           
        # Set to IMU mode (accelerometer + gyroscope, no magnetometer)
        self.get_logger().info("Setting IMU mode...")
        imu_mode = 0b00001000
        self.safe_write_register(opr_mode_addr, imu_mode)
        sleep(0.2)
        
        # Verify final mode
        mode = self.safe_read_register(opr_mode_addr)
        self.get_logger().info(f"Final IMU mode: 0x{mode:02X} ({mode:08b})")
        
        if mode != imu_mode:
            self.get_logger().warn(f"Mode mismatch! Expected 0x{imu_mode:02X}, got 0x{mode:02X}")
        
        unit_sel = self.safe_read_register(unit_sel_addr)
        self.get_logger().info(f"Final unit selection: 0x{unit_sel:02X} ({unit_sel:08b})")
        
        self.get_logger().info("IMU reset complete!")
    
    def combine_lsb_msb(self, lsb, msb):
        """Combine LSB and MSB into signed 16-bit integer"""
        value = (msb << 8) | lsb
        if value >= 32768:
            value -= 65536
        return value
    
    def imu_streaming(self):
        """Background thread for continuous IMU data reading"""
        self.get_logger().info("IMU streaming thread started")
        while rclpy.ok() and self.is_initialized:
            try:
                imu = self.get_imu_data()
                with self.data_lock:
                    self.imu_data = imu
            except Exception as e:
                self.get_logger().error(f"Error reading IMU data: {e}")
                sleep(0.1)  # Back off on error
                continue
            sleep(1/100)  # 100 Hz reading rate
        
    def get_imu_data(self):
        """Read IMU data from BNO055 registers"""
        # Gyroscope (rad/s) - convert from 1 LSB = 1/900 rad/s
        gyr_x = self.combine_lsb_msb(self.bno_usb_stick.read_register(0x14), self.bno_usb_stick.read_register(0x15)) / 900.0
        gyr_y = self.combine_lsb_msb(self.bno_usb_stick.read_register(0x16), self.bno_usb_stick.read_register(0x17)) / 900.0
        gyr_z = self.combine_lsb_msb(self.bno_usb_stick.read_register(0x18), self.bno_usb_stick.read_register(0x19)) / 900.0
        
        # Quaternion - convert from 1 LSB = 1/16384
        qua_w = self.combine_lsb_msb(self.bno_usb_stick.read_register(0x20), self.bno_usb_stick.read_register(0x21)) / 16384.0
        qua_x = self.combine_lsb_msb(self.bno_usb_stick.read_register(0x22), self.bno_usb_stick.read_register(0x23)) / 16384.0
        qua_y = self.combine_lsb_msb(self.bno_usb_stick.read_register(0x24), self.bno_usb_stick.read_register(0x25)) / 16384.0
        qua_z = self.combine_lsb_msb(self.bno_usb_stick.read_register(0x26), self.bno_usb_stick.read_register(0x27)) / 16384.0
        
        # Linear acceleration (m/s²) - convert from 1 LSB = 0.01 m/s²
        lin_x = self.combine_lsb_msb(self.bno_usb_stick.read_register(0x28), self.bno_usb_stick.read_register(0x29)) / 100.0
        lin_y = self.combine_lsb_msb(self.bno_usb_stick.read_register(0x2A), self.bno_usb_stick.read_register(0x2B)) / 100.0
        lin_z = self.combine_lsb_msb(self.bno_usb_stick.read_register(0x2C), self.bno_usb_stick.read_register(0x2D)) / 100.0
        
        return [gyr_x, gyr_y, gyr_z, qua_w, qua_x, qua_y, qua_z, lin_x, lin_y, lin_z]
            
    def timer_callback(self):
        """Publish IMU data at fixed rate"""
        if not self.is_initialized:
            return
            
        with self.data_lock:
            if self.imu_data is not None:
                imu = self.imu_data
                imu_msg = Imu()
                imu_msg.header.stamp = self.get_clock().now().to_msg()
                imu_msg.header.frame_id = 'imu_link'
                
                # Orientation (quaternion)
                imu_msg.orientation.w = imu[3]
                imu_msg.orientation.x = imu[4]
                imu_msg.orientation.y = imu[5]
                imu_msg.orientation.z = imu[6]
                
                # Angular velocity (rad/s)
                imu_msg.angular_velocity.x = imu[0]
                imu_msg.angular_velocity.y = imu[1]
                imu_msg.angular_velocity.z = imu[2]
                
                # Linear acceleration (m/s²)
                imu_msg.linear_acceleration.x = imu[7]
                imu_msg.linear_acceleration.y = imu[8]
                imu_msg.linear_acceleration.z = imu[9]
                
                self.publisher.publish(imu_msg)
    
def main(args=None):
    rclpy.init(args=args)
    try:
        node = BNO055USBSTICKNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()