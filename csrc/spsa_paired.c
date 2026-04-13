/*
 * Paired SPSA: the most efficient gradient descent possible for game engines.
 *
 * Key insight: play theta+delta vs theta-delta HEAD TO HEAD in the same game.
 * No baseline needed. No CRN overhead. Every game directly measures the
 * gradient direction. The paired outcome (who won) IS the gradient signal.
 *
 * Each iteration:
 *   1. Random perturbation direction (Rademacher: +1/-1 per param)
 *   2. Play 1008 4p games: 2 seats theta+delta, 2 seats theta-delta
 *   3. Count wins for each side
 *   4. gradient = (plus_wins - minus_wins) / games * sign_vector
 *   5. theta += lr * gradient
 *
 * This is ~2x more efficient than CRN-SPSA because every game measures
 * gradient directly, no wasted baseline games.
 *
 * 18 threads. ~3000 games/sec. 1000 games/iter = 0.3s/iter.
 * 10 minutes = 2000 iterations = 2M games.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"
#include "value.h"

#define NT 18
#define NUM_W 11
#define GAMES_PER_ITER 1008
#define GPT (GAMES_PER_ITER / NT)
#define AB_DEPTH 2
#define RUN_SECONDS 600

static const double ORIG[NUM_W] = {3e14,1e8,-1e8,1.,1e3,10.,1e2,1.,-5.,10.,10.1};
static const double DP[13] = {0,0,1./36,2./36,3./36,4./36,5./36,6./36,5./36,4./36,3./36,2./36,1./36};
static const double VB = 4.*(2.778/100.);
static const char *WN[NUM_W] = {"vps","prod","eprod","tiles","buildable","road","synergy","hand","discard","devs","army"};

/* Perturbation size: 0.5% of |param| */
static const double DELTA_PCT = 0.005;

static double theta[NUM_W];
static double plus_w[NUM_W], minus_w[NUM_W];
static volatile int running = 1;
static uint64_t run_seed_base;
static FILE *log_fp = NULL;

typedef struct { double W[4][NUM_W]; } GameWeights;

static double eval_w(Game *g, Color c, const double *W) {
    State *s=&g->state;int idx=s->color_to_index[(int)c];
    Board *b=&s->board;CatanMap *map=b->map;Coordinate robber=b->robber_coordinate;
    double rp[5]={0};
    for(int si=0;si<s->settlement_count[idx];si++){int node=s->settlements[idx][si];
        if(node<0||node>=NUM_NODES)continue;
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
            if(t2<0||t2>=NUM_LAND_TILES)continue;
            LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
            if(coord_eq(map->land_tile_coords[t2],robber))continue;rp[(int)t->resource]+=DP[t->number];}}
    for(int ci=0;ci<s->city_count[idx];ci++){int node=s->cities[idx][ci];
        if(node<0||node>=NUM_NODES)continue;
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
            if(t2<0||t2>=NUM_LAND_TILES)continue;
            LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
            if(coord_eq(map->land_tile_coords[t2],robber))continue;rp[(int)t->resource]+=2.*DP[t->number];}}
    double tp=0;int var=0;for(int r=0;r<5;r++){tp+=rp[r];if(rp[r]>0)var++;}
    double prod=tp+var*VB;
    Color enemy=COLOR_NONE;for(int i=0;i<s->num_players;i++)if(s->colors[i]!=c){enemy=s->colors[i];break;}
    double ep=0;
    if(enemy!=COLOR_NONE){int ei=s->color_to_index[(int)enemy];
        for(int si=0;si<s->settlement_count[ei];si++){int node=s->settlements[ei][si];
            if(node<0||node>=NUM_NODES)continue;
            for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
                if(t2<0||t2>=NUM_LAND_TILES)continue;
                LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(map->land_tile_coords[t2],robber))continue;ep+=DP[t->number];}}
        for(int ci=0;ci<s->city_count[ei];ci++){int node=s->cities[ei][ci];
            if(node<0||node>=NUM_NODES)continue;
            for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
                if(t2<0||t2>=NUM_LAND_TILES)continue;
                LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(map->land_tile_coords[t2],robber))continue;ep+=2.*DP[t->number];}}}
    int *ps=s->player_state[idx];
    int wh=ps[PS_WHEAT_IN_HAND],or2=ps[PS_ORE_IN_HAND],sh=ps[PS_SHEEP_IN_HAND],
        br=ps[PS_BRICK_IN_HAND],wo=ps[PS_WOOD_IN_HAND];
    double dc=(fmax(2-wh,0)+fmax(3-or2,0))/5.;double ds=(fmax(1-wh,0)+fmax(1-sh,0)+fmax(1-br,0)+fmax(1-wo,0))/4.;
    double syn=(2-dc-ds)/2.;int nih=wo+br+sh+wh+or2;
    bool ts[NUM_LAND_TILES]={0};int nt=0;
    for(int si=0;si<s->settlement_count[idx];si++){int node=s->settlements[idx][si];
        if(node<0||node>=NUM_NODES)continue;
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(t2>=0&&t2<NUM_LAND_TILES&&!ts[t2]){ts[t2]=1;nt++;}}}
    for(int ci=0;ci<s->city_count[idx];ci++){int node=s->cities[idx][ci];
        if(node<0||node>=NUM_NODES)continue;
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(t2>=0&&t2<NUM_LAND_TILES&&!ts[t2]){ts[t2]=1;nt++;}}}
    uint64_t reach[2]={0,0};
    for(int i=0;i<s->board.cc_count[(int)c];i++)bs_or(reach,reach,s->board.cc_sets[(int)c][i]);
    uint64_t avail[2];bs_and(avail,reach,s->board.buildable);
    int nb=__builtin_popcountll(avail[0])+__builtin_popcountll(avail[1]);
    double lrf=(nb==0)?W[5]:0.1;
    int nd=ps[PS_KNIGHT_IN_HAND]+ps[PS_YEAR_OF_PLENTY_IN_HAND]+ps[PS_MONOPOLY_IN_HAND]
          +ps[PS_ROAD_BUILDING_IN_HAND]+ps[PS_VICTORY_POINT_IN_HAND];
    return ps[PS_VICTORY_POINTS]*W[0]+prod*W[1]+ep*W[2]+nt*W[3]+nb*W[4]
          +ps[PS_LONGEST_ROAD_LENGTH]*lrf+syn*W[6]+nih*W[7]+(nih>7?W[8]:0)+nd*W[9]+ps[PS_PLAYED_KNIGHT]*W[10];
}

/* Game eval: seats 0,1 use plus_w, seats 2,3 use minus_w (from eval_ctx) */
static double paired_eval(Game *g, Color c) {
    int idx = g->state.color_to_index[(int)c];
    if (idx < 0 || idx >= 4 || !g->eval_ctx) return base_value_fn(g, c);
    return eval_w(g, c, ((const GameWeights *)g->eval_ctx)->W[idx]);
}

typedef struct {
    int tid, n;
    uint64_t seed_base;
    int plus_wins, minus_wins;
} WorkerArgs;

static void *worker(void *arg) {
    WorkerArgs *wa = (WorkerArgs*)arg;
    wa->plus_wins = 0; wa->minus_wins = 0;
    RngState rng; SearchCtx ctx;
    Color colors[4] = {COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};

    for (int gi = 0; gi < wa->n; gi++) {
        /* Rotate: even games plus=seats{0,1}, odd games plus=seats{2,3} */
        GameWeights gw;
        int plus_seats[2], minus_seats[2];
        if (gi % 2 == 0) {
            plus_seats[0]=0; plus_seats[1]=1; minus_seats[0]=2; minus_seats[1]=3;
        } else {
            plus_seats[0]=2; plus_seats[1]=3; minus_seats[0]=0; minus_seats[1]=1;
        }
        for (int s = 0; s < 4; s++) {
            int is_plus = (s == plus_seats[0] || s == plus_seats[1]);
            memcpy(gw.W[s], is_plus ? plus_w : minus_w, sizeof(double)*NUM_W);
        }

        uint64_t seed = wa->seed_base + (uint64_t)wa->tid * 100000ULL + gi;
        rng_init(&rng, seed);
        CatanMap map; build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &rng);
        Game game; game_init_with_map(&game, &map, 4, colors, seed, 7, false, 10);
        game.eval_ctx = &gw;
        Action acts[MAX_ACTIONS];
        int n = generate_playable_actions(&game.state, acts, MAX_ACTIONS);
        while (game_winning_color(&game)==COLOR_NONE && game.state.num_turns<TURNS_LIMIT) {
            Action a; if (n==1) a=acts[0];
            else { Color cur=state_current_color(&game.state);
                ctx.depth_counter=0; Game cp; game_copy(&cp,&game); cp.eval_ctx=&gw;
                SearchResult sr=alphabeta_search(&ctx,&cp,acts,n,AB_DEPTH,-1e30,1e30,cur,paired_eval);
                a=(sr.action.type||sr.action.color)?sr.action:acts[0]; }
            game_execute(&game,a,acts,&n);
        }
        Color w = game_winning_color(&game);
        if (w != COLOR_NONE) {
            int wi = game.state.color_to_index[(int)w];
            if (wi == plus_seats[0] || wi == plus_seats[1]) wa->plus_wins++;
            else wa->minus_wins++;
        }
    }
    return NULL;
}

/* HTTP server */
static char *dash_html = NULL; static long dash_len = 0;
static void load_dash(void) {
    FILE *f=fopen("spsa_dashboard.html","r"); if(!f)return;
    fseek(f,0,SEEK_END);dash_len=ftell(f);fseek(f,0,SEEK_SET);
    dash_html=malloc(dash_len+1);fread(dash_html,1,dash_len,f);dash_html[dash_len]=0;fclose(f);
}
static void *http_server(void *arg) {
    (void)arg;
    int fd=socket(AF_INET,SOCK_STREAM,0);
    int opt=1;setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
    struct sockaddr_in addr={.sin_family=AF_INET,.sin_addr.s_addr=INADDR_ANY,.sin_port=htons(8080)};
    if(bind(fd,(struct sockaddr*)&addr,sizeof(addr))<0){fprintf(stderr,"Port 8080 busy\n");return NULL;}
    listen(fd,16);fprintf(stderr,"Dashboard: http://localhost:8080\n");
    while(running){
        struct sockaddr_in cl;socklen_t cll=sizeof(cl);
        int c=accept(fd,(struct sockaddr*)&cl,&cll);if(c<0)continue;
        char buf[4096];int nr=read(c,buf,sizeof(buf)-1);if(nr<=0){close(c);continue;}buf[nr]=0;
        if(strstr(buf,"GET /data")){
            FILE *f=fopen("spsa_log.jsonl","r");
            if(f){fseek(f,0,SEEK_END);long sz=ftell(f);fseek(f,0,SEEK_SET);
                char *d=malloc(sz+1);fread(d,1,sz,f);d[sz]=0;fclose(f);
                char h[256];snprintf(h,256,"HTTP/1.1 200 OK\r\nContent-Type:application/json\r\nAccess-Control-Allow-Origin:*\r\nContent-Length:%ld\r\n\r\n",sz);
                write(c,h,strlen(h));write(c,d,sz);free(d);}
        } else if(dash_html){
            char h[256];snprintf(h,256,"HTTP/1.1 200 OK\r\nContent-Type:text/html\r\nContent-Length:%ld\r\n\r\n",dash_len);
            write(c,h,strlen(h));write(c,dash_html,dash_len);
        }
        close(c);
    }
    close(fd);return NULL;
}

int main(void) {
    run_seed_base = (uint64_t)time(NULL);
    srand((unsigned)run_seed_base);
    signal(SIGPIPE, SIG_IGN);

    RngState tmp; rng_init(&tmp,0);
    CatanMap tmp_map; build_map(&tmp_map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL,&tmp);
    board_init_static_graph(&tmp_map);

    load_dash();
    pthread_t http_th; pthread_create(&http_th,NULL,http_server,NULL);

    memcpy(theta, ORIG, sizeof(ORIG));
    log_fp = fopen("spsa_log.jsonl", "w");

    struct timespec T0,T1; clock_gettime(CLOCK_MONOTONIC,&T0);
    int total_games=0, iter=0;

    /* Learning rate per param: scale by 1/|param| so all move at similar % rate */
    double lr[NUM_W];
    for (int i=0;i<NUM_W;i++) lr[i] = fabs(ORIG[i]) * 0.0001; /* 0.01% step per unit gradient */

    printf("=== Paired SPSA Gradient Descent ===\n");
    printf("Perturbation: +/-%.1f%% | LR: 0.01%%/unit | %d games/iter | %d threads\n",
           DELTA_PCT*100, GAMES_PER_ITER, NT);
    printf("Dashboard: http://localhost:8080\n\n");

    while (1) {
        clock_gettime(CLOCK_MONOTONIC,&T1);
        double elapsed=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)/1e9;
        if (elapsed >= RUN_SECONDS) break;
        iter++;

        /* Random perturbation direction */
        int signs[NUM_W];
        for (int i=0;i<NUM_W;i++) signs[i] = (rand()%2)*2-1;

        /* Compute plus and minus weight vectors */
        for (int i=0;i<NUM_W;i++) {
            double delta = fabs(theta[i]) * DELTA_PCT * signs[i];
            plus_w[i]  = theta[i] + delta;
            minus_w[i] = theta[i] - delta;
        }

        /* Play head-to-head */
        uint64_t seed = run_seed_base * 1000003ULL + (uint64_t)iter * 131071ULL;
        pthread_t threads[NT]; WorkerArgs wa[NT];
        for (int i=0;i<NT;i++) {
            wa[i].tid=i; wa[i].n=GPT; wa[i].seed_base=seed;
        }
        for (int i=0;i<NT;i++) pthread_create(&threads[i],NULL,worker,&wa[i]);
        for (int i=0;i<NT;i++) pthread_join(threads[i],NULL);

        int pw=0, mw=0;
        for (int i=0;i<NT;i++) { pw+=wa[i].plus_wins; mw+=wa[i].minus_wins; }
        total_games += GAMES_PER_ITER;

        double plus_wr = (double)pw/(pw+mw+1e-10);
        double gradient_signal = plus_wr - 0.5; /* >0 means plus is better */

        /* Decay schedule */
        double decay = 1.0 / pow(iter + 100, 0.602);

        /* Update theta */
        for (int i=0;i<NUM_W;i++)
            theta[i] += decay * lr[i] * gradient_signal * signs[i];

        /* Log */
        clock_gettime(CLOCK_MONOTONIC,&T1);
        elapsed=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)/1e9;
        double gps = total_games/elapsed;

        fprintf(log_fp,"{\"iter\":%d,\"pw\":%d,\"mw\":%d,\"grad\":%.4f,\"elapsed\":%.1f,\"games\":%d,\"gps\":%.0f,\"w\":[",
                iter,pw,mw,gradient_signal,elapsed,total_games,gps);
        for(int i=0;i<NUM_W;i++) fprintf(log_fp,"%s%.6g",i?",":"",theta[i]);
        fprintf(log_fp,"],\"pct\":[");
        for(int i=0;i<NUM_W;i++){double p=(theta[i]-ORIG[i])/fabs(ORIG[i])*100;fprintf(log_fp,"%s%.4f",i?",":"",p);}
        fprintf(log_fp,"]}\n"); fflush(log_fp);

        if (iter%50==0) {
            printf("iter %4d: +%d -%d grad=%+.3f [%dk,%.0fs,%.0f g/s] | ",
                   iter,pw,mw,gradient_signal,total_games/1000,elapsed,gps);
            for(int i=0;i<NUM_W;i++){double p=(theta[i]-ORIG[i])/fabs(ORIG[i])*100;
                if(fabs(p)>0.01)printf("%s%+.3f%% ",WN[i],p);}
            printf("\n");
        }
    }

    clock_gettime(CLOCK_MONOTONIC,&T1);
    double elapsed=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)/1e9;
    printf("\n=== Done: %d iters, %dk games, %.0fs, %.0f g/s ===\n",
           iter,total_games/1000,elapsed,total_games/elapsed);
    printf("Final theta (drift from baseline):\n");
    for(int i=0;i<NUM_W;i++){double p=(theta[i]-ORIG[i])/fabs(ORIG[i])*100;
        printf("  %-12s = %15.6g  (%+.4f%%)\n",WN[i],theta[i],p);}

    if(log_fp)fclose(log_fp);
    printf("\nDashboard: http://localhost:8080 (Ctrl-C to quit)\n");
    running=0; sleep(3600);
    return 0;
}
