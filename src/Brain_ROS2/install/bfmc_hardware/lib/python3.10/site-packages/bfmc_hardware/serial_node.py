import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import serial
import traceback
import threading
import time

# CONFIGURATION
SERIAL_PORT = '/dev/ttyACM0'  
BAUD_RATE   = 115200          
CAMERA_ID   = 0     

class TestSerial:
    def __init__(self, logger):
        self.logger = logger
        self.is_open = True
        self.logger.warn("!!! CURRENTLY RUNNING IN DEBUG MODE (NO NUCLEO CONNECTED) !!!")
        
    def write(self, data):
        try:
            decoded_str = data.decode('utf-8').strip()
            self.logger.info(f"[text send] >> {decoded_str}")
        except:
            pass
        
    def readline(self):
        time.sleep(1)
        return b""
    
    def flush(self):
        pass
    
    def close(self):
        self.is_open = False
        self.logger.info("Test serial closed")

class SerialBridgeNode(Node):
    def __init__(self):
        super().__init__("serial_node")
                
        self.sub_cmd = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        self.pub_serial_read = self.create_publisher(String, '/serial/read', 10)
        
        self.serial_conn = None
        self.connect_serial()
        
        self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.read_thread.start()
        
        self.get_logger().info("Serial Bridge Ready")
        
    def connect_serial(self):
        port = SERIAL_PORT
        baud = BAUD_RATE
        try:
            self.serial_conn = serial.Serial(port, baud, timeout=1)
            self.serial_conn.flush()
            self.get_logger().info(f"Connected to Nucleo on {port}")
        except Exception as e:
            self.get_logger().error(f"Cannot connect to Nucleo: {e}")
            self.get_logger().info("Switching to debug (test) mode...")
            self.serial_conn = TestSerial(self.get_logger())
            
    def cmd_vel_callback(self, msg):
        if self.serial_conn and self.serial_conn.is_open:
            speed = msg.linear.x
            steer = msg.angular.z
            
            cmd_str = f"#vcdCalib:{speed:.2f};{steer:.2f};100;;\r\n"
            
            try:
                self.serial_conn.write(cmd_str.encode('utf-8'))
            except Exception as e:
                self.get_logger().warn(f"Write error: {e}")
                
    def read_loop(self):
        while rclpy.ok():
            if self.serial_conn and self.serial_conn.is_open:
                try:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    if line:
                        msg = String()
                        msg.data = line
                        self.pub_serial_read.publish(msg)
                except Exception:
                    pass
            else:
                time.sleep(1)

    def stop_robot(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write("#vcdCalib:0.0;0.0;0;;\r\n".encode("utf-8"))
            self.serial_conn.close()
            
def main(args = None):
    rclpy.init(args=args)
    node = SerialBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    main()        