# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from . import PointCloud_pb2 as PointCloud__pb2


class MainServerStub(object):
    """responce server
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.get_point_cloud = channel.unary_stream(
            '/pointcloud.MainServer/get_point_cloud',
            request_serializer=PointCloud__pb2.PointRequest.SerializeToString,
            response_deserializer=PointCloud__pb2.PointReply.FromString,
        )


class MainServerServicer(object):
    """responce server
    """

    def get_point_cloud(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MainServerServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'get_point_cloud': grpc.unary_stream_rpc_method_handler(
            servicer.get_point_cloud,
            request_deserializer=PointCloud__pb2.PointRequest.FromString,
            response_serializer=PointCloud__pb2.PointReply.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'pointcloud.MainServer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))

 # This class is part of an EXPERIMENTAL API.


class MainServer(object):
    """responce server
    """

    @staticmethod
    def get_point_cloud(request,
                        target,
                        options=(),
                        channel_credentials=None,
                        call_credentials=None,
                        compression=None,
                        wait_for_ready=None,
                        timeout=None,
                        metadata=None):
        return grpc.experimental.unary_stream(request, target, '/pointcloud.MainServer/get_point_cloud',
                                              PointCloud__pb2.PointRequest.SerializeToString,
                                              PointCloud__pb2.PointReply.FromString,
                                              options, channel_credentials,
                                              call_credentials, compression, wait_for_ready, timeout, metadata)
