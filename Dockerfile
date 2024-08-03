FROM python:3.11

WORKDIR /fl_attacks

COPY ./ ./

RUN make install

CMD [ "make", "run_adult_back_krum" ]