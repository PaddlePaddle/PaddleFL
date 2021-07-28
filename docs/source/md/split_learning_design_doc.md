# Split Learning Design Doc

## Background

SplitNN enables different data holders (usually with different profile) to train model cooperatively, without revealing their own data. In each training epoch, participants train and update their own part of network locally, and communicate when necessary. In the whole process, intermediate variate are encrypted before transmission. SplitNN only encrypt hidden vector which need to communicate with client, save lot of time.

We choose SplitNN because following reasons:

1. SplitNN has higher efficiency: within the same time we can train more dataset.
2. SplitNN support more model: all paddlepaddle models are theoretically supported.
3. SplitNN support multi training framework: intermediate variate communicated by grpc protocol and training process is decoupled.

## Design

Our solution consists of three parts:
1. Client SDK
2. Server
3. Profile Service

<img src='../../../images/PFS-design.png' align="middle"/>

### Client SDK
Client SDK encrypt Unique User Identify(UUID), push to server, get hidden vector *h2* with noisy(for protect server's data privacy), then concat *h2* with self *h1* as *h3*, to infer result *y*. For execute backward network, calculate server's variable gradient, to update server's paramter, Client SDK will push *Δh2* back to server.

Client SDK now support following features:
1. Multi language: based on GRPC.
2. Build self-defined network: support static graph mode and dynamic graph mode, split entire network by specific key.
3. Sparse feature extract(TODO).

### Server
Server get encrypted ID, then use collision algorithm get profile. Server will execute forward network, calculate hidden vector *h2*, and push noised *h2* to client, then wait for client response, finish remain backward network in server.

Server now support following features:
1. Use encrypted ID as key, get profile from key-value service.
2. Support GRPC.
3. Build self-defined network: support static graph mode and dynamic graph mode, split entire network by specific key.
4. Support noise *h2* to protect user's privacy.
5. Support split network and separate execute.
6. Support distributed execute(TODO).

### Profile Service
Traditional profile prepare need O(n) time to filte user profile, which n usually near 1e9.

For quickly get profile for server training, we construe a TB level profile service. Now we update user profile at hourly granularity, when client use encrypted ID, service can hit serveral user primary key and get multi-faceted user profile in O(1) time.
