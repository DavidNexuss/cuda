typedef struct _Renderer Renderer;
typedef struct _Scene    Scene;

typedef struct {
  int width;
  int height;
} RendererDesc;

Renderer* rendererCreate(RendererDesc desc);
void      rendererUpload(Renderer* renderer, Scene* scene);
void      rendererDestoy(Renderer* renderer);
void      rendererDraw(Renderer* renderer, Scene* scene);
int       rendererPollEvents(Renderer* renderer);
