using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

namespace UndreamAI.LlamaLib
{
    // Data structures for LoRA operations
    public struct LoraIdScale
    {
        public int Id { get; set; }
        public float Scale { get; set; }

        public LoraIdScale(int id, float scale)
        {
            Id = id;
            Scale = scale;
        }
    }

    public struct LoraIdScalePath
    {
        public int Id { get; set; }
        public float Scale { get; set; }
        public string Path { get; set; }

        public LoraIdScalePath(int id, float scale, string path)
        {
            Id = id;
            Scale = scale;
            Path = path;
        }
    }

    // Base LLM class
    public abstract class LLM : IDisposable
    {
        public LlamaLib llamaLib = null;
        public IntPtr llm = IntPtr.Zero;
        protected readonly object _disposeLock = new object();
        public bool disposed = false;

        protected LLM() { }

        protected LLM(LlamaLib llamaLibInstance)
        {
            llamaLib =
                llamaLibInstance ?? throw new ArgumentNullException(nameof(llamaLibInstance));
        }

        public static void Debug(int debugLevel)
        {
            LlamaLib.Debug(debugLevel);
        }

        public static void LoggingCallback(LlamaLib.CharArrayCallback callback)
        {
            LlamaLib.LoggingCallback(callback);
        }

        public static void LoggingStop()
        {
            LlamaLib.LoggingStop();
        }

        protected void CheckLlamaLib()
        {
            if (disposed)
                throw new ObjectDisposedException(GetType().Name);
            if (llamaLib == null)
                throw new InvalidOperationException("LlamaLib instance is not initialized");
            if (llm == IntPtr.Zero)
                throw new InvalidOperationException("LLM instance is not initialized");
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            disposed = true;
        }

        ~LLM()
        {
            Dispose(false);
        }

        public string ApplyTemplate(JArray messages = null)
        {
            if (messages == null)
                throw new ArgumentNullException(nameof(messages));
            CheckLlamaLib();
            IntPtr result = llamaLib.LLM_Apply_Template(llm, messages.ToString() ?? string.Empty);
            return llamaLib.PtrToStringAndFree(result);
        }

        public List<int> Tokenize(string content)
        {
            if (string.IsNullOrEmpty(content))
                throw new ArgumentNullException(nameof(content));

            CheckLlamaLib();
            IntPtr result = llamaLib.LLM_Tokenize(llm, content ?? string.Empty);
            string resultStr = llamaLib.PtrToStringAndFree(result);
            List<int> ret = new List<int>();
            try
            {
                JArray json = JArray.Parse(resultStr);
                ret = json?.ToObject<List<int>>();
            }
            catch { }
            return ret;
        }

        public string Detokenize(List<int> tokens)
        {
            if (tokens == null)
                throw new ArgumentNullException(nameof(tokens));

            CheckLlamaLib();
            JArray tokensJSON = JArray.FromObject(tokens);
            IntPtr result = llamaLib.LLM_Detokenize(llm, tokensJSON.ToString() ?? string.Empty);
            return llamaLib.PtrToStringAndFree(result);
        }

        public string Detokenize(int[] tokens)
        {
            if (tokens == null)
                throw new ArgumentNullException(nameof(tokens));
            return Detokenize(new List<int>(tokens));
        }

        public List<float> Embeddings(string content)
        {
            if (string.IsNullOrEmpty(content))
                throw new ArgumentNullException(nameof(content));

            CheckLlamaLib();

            IntPtr result = llamaLib.LLM_Embeddings(llm, content ?? string.Empty);
            string resultStr = llamaLib.PtrToStringAndFree(result);

            List<float> ret = new List<float>();
            try
            {
                JArray json = JArray.Parse(resultStr);
                ret = json?.ToObject<List<float>>();
            }
            catch { }
            return ret;
        }

        public void SetCompletionParameters(JObject parameters = null)
        {
            CheckLlamaLib();
            llamaLib.LLM_Set_Completion_Parameters(llm, parameters?.ToString() ?? string.Empty);
        }

        public JObject GetCompletionParameters()
        {
            CheckLlamaLib();
            JObject parameters = new JObject();
            IntPtr result = llamaLib.LLM_Get_Completion_Parameters(llm);
            string parametersString = llamaLib.PtrToStringAndFree(result);
            if (string.IsNullOrEmpty(parametersString))
                parametersString = "{}";
            try
            {
                parameters = JObject.Parse(parametersString);
            }
            catch { }
            return parameters;
        }

        public void SetGrammar(string grammar)
        {
            CheckLlamaLib();
            llamaLib.LLM_Set_Grammar(llm, grammar ?? string.Empty);
        }

        public string GetGrammar()
        {
            CheckLlamaLib();
            IntPtr result = llamaLib.LLM_Get_Grammar(llm);
            return llamaLib.PtrToStringAndFree(result);
        }

        public void CheckCompletionInternal(string prompt)
        {
            if (string.IsNullOrEmpty(prompt))
                throw new ArgumentNullException(nameof(prompt));
            CheckLlamaLib();
        }

        public string CompletionInternal(
            string prompt,
            LlamaLib.CharArrayCallback callback,
            int idSlot
        )
        {
            IntPtr result = llamaLib.LLM_Completion(llm, prompt ?? string.Empty, callback, idSlot);
            return llamaLib.PtrToStringAndFree(result);
        }

        public string Completion(
            string prompt,
            LlamaLib.CharArrayCallback callback = null,
            int idSlot = -1
        )
        {
            CheckCompletionInternal(prompt);
            return CompletionInternal(prompt, callback, idSlot);
        }

        public async Task<string> CompletionAsync(
            string prompt,
            LlamaLib.CharArrayCallback callback = null,
            int idSlot = -1,
            System.Threading.CancellationToken cancellationToken = default
        )
        {
            CheckCompletionInternal(prompt);
            if (cancellationToken.IsCancellationRequested)
                throw new OperationCanceledException(cancellationToken);

            using (
                cancellationToken.Register(() =>
                {
                    try
                    {
                        if (this is LLMLocal local)
                            local.Cancel(idSlot < 0 ? 0 : idSlot);
                    }
                    catch { }
                })
            )
            {
                string result = await Task.Run(
                        () => CompletionInternal(prompt, callback, idSlot),
                        cancellationToken
                    )
                    .ConfigureAwait(false);
                cancellationToken.ThrowIfCancellationRequested();
                return result;
            }
        }

#if NETSTANDARD2_1_OR_GREATER || NETCOREAPP3_0_OR_GREATER
        public async IAsyncEnumerable<string> CompletionStreamAsync(
            string prompt,
            int idSlot = -1,
            [System.Runtime.CompilerServices.EnumeratorCancellation]
                System.Threading.CancellationToken cancellationToken = default
        )
        {
            CheckCompletionInternal(prompt);
            var channel = System.Threading.Channels.Channel.CreateUnbounded<string>(
                new System.Threading.Channels.UnboundedChannelOptions
                {
                    SingleWriter = true,
                    SingleReader = true,
                }
            );

            LlamaLib.CharArrayCallback callback = (text) =>
            {
                channel.Writer.TryWrite(text);
            };

            var completionTask = Task.Run(
                () =>
                {
                    try
                    {
                        using (
                            cancellationToken.Register(() =>
                            {
                                try
                                {
                                    if (this is LLMLocal local)
                                        local.Cancel(idSlot < 0 ? 0 : idSlot);
                                }
                                catch { }
                            })
                        )
                        {
                            CompletionInternal(prompt, callback, idSlot);
                        }
                    }
                    finally
                    {
                        channel.Writer.Complete();
                    }
                },
                cancellationToken
            );

            while (await channel.Reader.WaitToReadAsync(cancellationToken).ConfigureAwait(false))
            {
                while (channel.Reader.TryRead(out var chunk))
                {
                    yield return chunk;
                }
            }

            await completionTask.ConfigureAwait(false);
        }
#endif

        public async Task<List<int>> TokenizeAsync(
            string content,
            System.Threading.CancellationToken cancellationToken = default
        )
        {
            return await Task.Run(() => Tokenize(content), cancellationToken).ConfigureAwait(false);
        }

        public async Task<string> DetokenizeAsync(
            List<int> tokens,
            System.Threading.CancellationToken cancellationToken = default
        )
        {
            return await Task.Run(() => Detokenize(tokens), cancellationToken)
                .ConfigureAwait(false);
        }

        public async Task<List<float>> EmbeddingsAsync(
            string content,
            System.Threading.CancellationToken cancellationToken = default
        )
        {
            return await Task.Run(() => Embeddings(content), cancellationToken)
                .ConfigureAwait(false);
        }

        public async Task<string> ApplyTemplateAsync(
            JArray messages,
            System.Threading.CancellationToken cancellationToken = default
        )
        {
            return await Task.Run(() => ApplyTemplate(messages), cancellationToken)
                .ConfigureAwait(false);
        }
    }

    // LLMLocal class
    public abstract class LLMLocal : LLM
    {
        protected LLMLocal()
            : base() { }

        protected LLMLocal(LlamaLib llamaLibInstance)
            : base(llamaLibInstance) { }

        public virtual string SaveSlot(int idSlot, string filepath)
        {
            if (string.IsNullOrEmpty(filepath))
                throw new ArgumentNullException(nameof(filepath));

            IntPtr result = llamaLib.LLM_Save_Slot(llm, idSlot, filepath ?? string.Empty);
            return llamaLib.PtrToStringAndFree(result);
        }

        public virtual string LoadSlot(int idSlot, string filepath)
        {
            if (string.IsNullOrEmpty(filepath) || !File.Exists(filepath))
                throw new ArgumentNullException(nameof(filepath));

            IntPtr result = llamaLib.LLM_Load_Slot(llm, idSlot, filepath ?? string.Empty);
            return llamaLib.PtrToStringAndFree(result);
        }

        public virtual void Cancel(int idSlot)
        {
            CheckLlamaLib();
            llamaLib.LLM_Cancel(llm, idSlot);
        }

        public async Task<string> SaveSlotAsync(
            int idSlot,
            string filepath,
            System.Threading.CancellationToken cancellationToken = default
        )
        {
            return await Task.Run(() => SaveSlot(idSlot, filepath), cancellationToken)
                .ConfigureAwait(false);
        }

        public async Task<string> LoadSlotAsync(
            int idSlot,
            string filepath,
            System.Threading.CancellationToken cancellationToken = default
        )
        {
            return await Task.Run(() => LoadSlot(idSlot, filepath), cancellationToken)
                .ConfigureAwait(false);
        }
    }

    // LLMProvider class
    public abstract class LLMProvider : LLMLocal
    {
        protected LLMProvider()
            : base() { }

        protected LLMProvider(LlamaLib llamaLibInstance)
            : base(llamaLibInstance) { }

        public void EnableReasoning(bool enableReasoning)
        {
            CheckLlamaLib();
            llamaLib.LLM_Enable_Reasoning(llm, enableReasoning);
        }

        // LoRA Weight methods
        public string BuildLoraWeightJSON(List<LoraIdScale> loras)
        {
            var jsonArray = new JArray();
            foreach (var lora in loras)
            {
                jsonArray.Add(new JObject { ["id"] = lora.Id, ["scale"] = lora.Scale });
            }
            return jsonArray.ToString();
        }

        public bool LoraWeight(List<LoraIdScale> loras)
        {
            if (loras == null)
                throw new ArgumentNullException(nameof(loras));

            var lorasJSON = BuildLoraWeightJSON(loras);
            return llamaLib.LLM_Lora_Weight(llm, lorasJSON ?? string.Empty);
        }

        public bool LoraWeight(params LoraIdScale[] loras)
        {
            return LoraWeight(new List<LoraIdScale>(loras));
        }

        // LoRA List methods
        public List<LoraIdScalePath> ParseLoraListJSON(string result)
        {
            var loras = new List<LoraIdScalePath>();
            try
            {
                var jsonArray = JArray.Parse(result);
                foreach (var item in jsonArray)
                {
                    int id = item["id"]?.ToObject<int>() ?? -1;
                    if (id < 0)
                        continue;
                    loras.Add(
                        new LoraIdScalePath(
                            id,
                            item["scale"]?.ToObject<float>() ?? 0.0f,
                            item["path"]?.ToString() ?? string.Empty
                        )
                    );
                }
            }
            catch { }
            return loras;
        }

        public string LoraListJSON()
        {
            CheckLlamaLib();
            var result = llamaLib.LLM_Lora_List(llm);
            return llamaLib.PtrToStringAndFree(result);
        }

        public List<LoraIdScalePath> LoraList()
        {
            var jsonResult = LoraListJSON();
            return ParseLoraListJSON(jsonResult);
        }

        // Server methods
        public bool Start()
        {
            CheckLlamaLib();
            llamaLib.LLM_Start(llm);
            return llamaLib.LLM_Started(llm);
        }

        public async Task<bool> StartAsync(
            System.Threading.CancellationToken cancellationToken = default
        )
        {
            CheckLlamaLib();
            return await Task.Run(
                    () =>
                    {
                        llamaLib.LLM_Start(llm);
                        return llamaLib.LLM_Started(llm);
                    },
                    cancellationToken
                )
                .ConfigureAwait(false);
        }

        public bool Started()
        {
            CheckLlamaLib();
            return llamaLib.LLM_Started(llm);
        }

        public void Stop()
        {
            CheckLlamaLib();
            llamaLib.LLM_Stop(llm);
        }

        public void StartServer(string host = "0.0.0.0", int port = -1, string apiKey = "")
        {
            CheckLlamaLib();
            host = string.IsNullOrEmpty(host) ? "0.0.0.0" : host;
            apiKey = apiKey ?? string.Empty;

            llamaLib.LLM_Start_Server(llm, host, port, apiKey);
        }

        public void StopServer()
        {
            CheckLlamaLib();
            llamaLib.LLM_Stop_Server(llm);
        }

        public void JoinService()
        {
            CheckLlamaLib();
            llamaLib.LLM_Join_Service(llm);
        }

        public void JoinServer()
        {
            CheckLlamaLib();
            llamaLib.LLM_Join_Server(llm);
        }

        public void SetSSL(string sslCert, string sslKey)
        {
            if (string.IsNullOrEmpty(sslCert))
                throw new ArgumentNullException(nameof(sslCert));
            if (string.IsNullOrEmpty(sslKey))
                throw new ArgumentNullException(nameof(sslKey));

            CheckLlamaLib();
            llamaLib.LLM_Set_SSL(llm, sslCert ?? string.Empty, sslKey ?? string.Empty);
        }

        public int EmbeddingSize()
        {
            CheckLlamaLib();
            return llamaLib.LLM_Embedding_Size(llm);
        }

        protected override void Dispose(bool disposing)
        {
            lock (_disposeLock)
            {
                if (!disposed)
                {
                    if (llm != IntPtr.Zero && llamaLib != null)
                    {
                        try
                        {
                            llamaLib.LLM_Delete(llm);
                        }
                        catch (Exception) { }
                        llm = IntPtr.Zero;
                    }
                    if (disposing)
                    {
                        llamaLib?.Dispose();
                        llamaLib = null;
                    }
                    disposed = true;
                }
            }
            base.Dispose(disposing);
        }
    }
}
