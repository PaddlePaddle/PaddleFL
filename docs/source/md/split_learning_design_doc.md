# Split Learning Design Doc

## Background

SplitNN will split entire network to multi part, each participant(at least 2) only calculate their own part. Compare with MPC, SplitNN only encrypt vector which used for communicate, save lot of time(but also reduce security)

We choose SplitNN because follow reasons:

1. SplitNN has higher efficiency: X% improvement over MPC.
2. Due to SplitNN has higher efficiency, within the same time we can train more dataset.
3. Splitnn support more model: MPC need to implement MPC OP, and MPC GPU OP is hard to implement.

## Design

Entire SplitNN consists of three parts:
1. Client SDK
2. Server
3. Profile Service

<img src='../../../images/PFS-design.png' width = "1000" height = "622" align="middle"/>

### Client SDK
Client SDK encrypt Unique User Identify(UUID), push to server, get hidden vector *h2* with noisy(for protect server privacy), then combine *h2* with self *h1* as *h3*, to infer result *y*. For execute backward network, calculate server's variable gradient, to update server's paramter, Client SDK will push *Δh2* to server.

Client SDK now support following feature:
1. Multi coding language: based on GRPC.
2. Build network: static graph and dynamic graph, split entire network by specific key.
3. Sparse feature extract(TODO).

### Server
Server get encrypted UUID, then use collision algorithm get profile. Server will execute forward network, calculate hidden vector *h2* with noisy, and push to client, then wait client response, finish entire server network.

Server now support following feature:
1. Use encrypted UUID as key, get profile from profile database.
2. Support GRPC.
3. Build network: static graph and dynamic graph, split entire network by specific key.
4. Support noise *h2* to protect user's privacy
5. Support split network and separate execute.
6. Support distributed execute(TODO).

### Profile Service
For quickly get profile for server training, we construe a TB level profile service. Now we update user profile at hourly granularity, when client use encrypted UUID, service can hit serveral user primary key and get multi-faceted user profile.
