/* eslint-disable import/no-extraneous-dependencies */
import {
  Candidate,
  Content,
  FinishReason,
  FunctionCallingConfigMode,
  GenerateContentParameters,
  GenerateContentResponse,
  GoogleGenAI,
  HttpOptions,
  Part,
  Schema,
  Tool,
  ToolConfig,
  UsageMetadata,
} from '@google/genai'
import { nanoid } from 'nanoid'
import OpenAI from 'openai'
import {
  ChatCompletionChunk,
  ChatCompletionContentPart,
} from 'openai/resources/index'
import { ChatCompletionMessageToolCall } from 'openai/src/resources/index.js'

import {
  GeminiModel,
  ProviderCompletionParams,
  RequestOptions,
} from '../chat/index.js'
import {
  CompletionResponse,
  StreamCompletionResponse,
} from '../userTypes/index.js'
import { BaseHandler } from './base.js'
import { InputError } from './types.js'
import {
  convertMessageContentToString,
  fetchThenParseImage,
  getTimestamp,
} from './utils.js'

export const convertContentsToParts = async (
  contents: Array<ChatCompletionContentPart> | string | null | undefined,
  systemPrefix: string
): Promise<Part[]> => {
  if (contents === null || contents === undefined) {
    return []
  }

  if (typeof contents === 'string') {
    return [
      {
        text: `${systemPrefix}${contents}`,
      },
    ]
  } else {
    const allParts: Promise<Part>[] = contents.map(async (part) => {
      if (part.type === 'text') {
        return {
          text: `${systemPrefix}${part.text}`,
        } as Part
      } else if (part.type === 'image_url') {
        const imageData = await fetchThenParseImage(part.image_url.url)
        return {
          inlineData: {
            mimeType: imageData.mimeType,
            data: imageData.content,
          },
        } satisfies Part
      } else {
        throw new InputError(
          `Invalid content part type: ${
            (part as any).type
          }. Must be "text" or "image_url".`
        )
      }
    })
    return Promise.all(allParts)
  }
}

// Google only supports the `model` and `user` roles, so we map everything else to `user`
// We handle the `system` role similarly to Anthropic where the first message is placed in the systemInstruction field
// if it is a system message, and the rest are treated as user messages but with a "System:" prefix.
export const convertRole = (
  role: 'function' | 'system' | 'user' | 'assistant' | 'tool'
) => {
  switch (role) {
    case 'assistant':
      return 'model'
    case 'function':
    case 'tool':
    case 'user':
    case 'system':
      return 'user'
    default:
      throw new InputError(`Unexpected message role: ${role}`)
  }
}

export const convertAssistantMessage = (
  message: OpenAI.Chat.Completions.ChatCompletionMessage
): Content => {
  const parts: Part[] = message.tool_calls
    ? message.tool_calls.map((call): Part => {
        return {
          functionCall: {
            name: call.function.name,
            args: JSON.parse(call.function.arguments),
          },
        }
      })
    : []

  if (message.content && message.content.length) {
    parts.push({
      text: message.content,
    })
  }

  return {
    role: convertRole(message.role),
    parts,
  }
}

export const convertMessageToContent = async (
  message: OpenAI.Chat.Completions.ChatCompletionMessageParam,
  includeSystemPrefix: boolean
): Promise<Content> => {
  switch (message.role) {
    case 'tool':
      const toolContent = convertMessageContentToString(message.content)
      let parsedResponse: any
      try {
        parsedResponse = JSON.parse(toolContent)
      } catch {
        parsedResponse = { result: toolContent, status: 'success' }
      }
      return {
        role: convertRole(message.role),
        parts: [
          {
            functionResponse: {
              name: message.tool_call_id,
              response: parsedResponse,
            },
          },
        ],
      }
    case 'assistant':
      // Casting to the ChatCompletionMessage type here is fine because the role will only ever be
      // assistant if the object is a ChatCompletionMessage
      return convertAssistantMessage(
        message as OpenAI.Chat.Completions.ChatCompletionMessage
      )
    case 'user':
    case 'system':
      const systemPrefix =
        message.role === 'system' && includeSystemPrefix ? 'System:\n' : ''
      return {
        role: convertRole(message.role),
        parts: await convertContentsToParts(message.content, systemPrefix),
      }
    default:
      throw new InputError(`Unexpected message role: ${message.role}`)
  }
}

export const convertMessagesToContents = async (
  messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[]
): Promise<{
  contents: Content[]
  systemInstruction: Content | undefined
}> => {
  const clonedMessages = structuredClone(messages)

  // Pop the first element from the user-defined `messages` array if it begins with a 'system'
  // message. The returned element will be used for Gemini's `systemInstruction` parameter. We only pop the
  // system message if it's the first element in the array so that the order of the messages remains
  // unchanged.
  let systemInstruction: Content | undefined
  if (clonedMessages.length > 0 && clonedMessages[0].role === 'system') {
    const systemMessage = clonedMessages.shift()
    systemInstruction =
      systemMessage !== undefined
        ? await convertMessageToContent(systemMessage, false)
        : undefined
  }

  const converted: Array<Content> = []
  for (const message of clonedMessages) {
    if (message.role === 'system' || message.role === 'user') {
      converted.push(await convertMessageToContent(message, true))
    } else if (message.role === 'assistant') {
      converted.push(await convertMessageToContent(message, true))
      if (message.tool_calls !== undefined) {
        for (const assistantToolCall of message.tool_calls) {
          const toolResult = clonedMessages.find(
            (m) => m.role === 'tool' && m.tool_call_id === assistantToolCall.id
          )
          if (toolResult === undefined) {
            throw new Error(
              `Could not find tool message with the id: ${assistantToolCall.id}`
            )
          }
          converted.push(await convertMessageToContent(toolResult, true))
        }
      }
    }
  }

  return {
    contents: converted,
    systemInstruction,
  }
}

export const convertFinishReason = (
  finishReason: FinishReason,
  parts: Part[] | undefined
): 'stop' | 'length' | 'tool_calls' | 'content_filter' | 'function_call' => {
  if (parts?.some((part) => part.functionCall !== undefined)) {
    return 'tool_calls'
  }

  switch (finishReason) {
    case FinishReason.STOP:
      return 'stop'
    case FinishReason.MAX_TOKENS:
      return 'length'
    case FinishReason.SAFETY:
      return 'content_filter'
    case FinishReason.OTHER:
    case FinishReason.FINISH_REASON_UNSPECIFIED:
    case FinishReason.RECITATION:
      return 'stop'
    default:
      return 'stop'
  }
}

export const convertToolCalls = (
  candidate: Candidate
): Array<ChatCompletionMessageToolCall> | undefined => {
  const toolCalls = candidate.content?.parts
    .filter((part) => part.functionCall !== undefined)
    .map((part, index) => {
      return {
        id: nanoid(),
        index,
        function: {
          arguments: JSON.stringify(part.functionCall!.args),
          name: part.functionCall!.name,
        },
        // Casting as 'function' just fixes a minor type issue
        type: 'function' as 'function',
      }
    })

  if (toolCalls !== undefined && toolCalls.length > 0) {
    return toolCalls
  } else {
    return undefined
  }
}

export const convertStreamToolCalls = (
  candidate: Candidate
): Array<ChatCompletionChunk.Choice.Delta.ToolCall> | undefined => {
  return convertToolCalls(candidate)?.map((toolCall, index) => {
    return {
      ...toolCall,
      index,
    }
  })
}

export const convertResponseMessage = (
  candidate: Candidate
): CompletionResponse['choices'][number]['message'] => {
  return {
    content: candidate.content?.parts.map((part) => part.text).join('') ?? null,
    role: 'assistant',
    tool_calls: convertToolCalls(candidate),
    refusal: null,
  }
}

export const convertUsageData = (
  usageMetadata: UsageMetadata
): CompletionResponse['usage'] => {
  return {
    completion_tokens: usageMetadata.totalTokenCount,
    prompt_tokens: usageMetadata.promptTokenCount,
    total_tokens: usageMetadata.totalTokenCount,
  }
}

export const convertToolConfig = (
  toolChoice:
    | OpenAI.Chat.Completions.ChatCompletionToolChoiceOption
    | undefined,
  tools: OpenAI.Chat.Completions.ChatCompletionTool[] | undefined
): ToolConfig => {
  // If tool choise is an object, then it is a required specific function
  if (typeof toolChoice === 'object') {
    return {
      functionCallingConfig: {
        mode: FunctionCallingConfigMode.ANY,
        allowedFunctionNames: [toolChoice.function.name],
      },
    }
  }

  switch (toolChoice) {
    case 'auto':
      return {
        functionCallingConfig: {
          mode: FunctionCallingConfigMode.AUTO,
        },
      }
    case 'none':
      return {
        functionCallingConfig: {
          mode: FunctionCallingConfigMode.NONE,
        },
      }
    case 'required':
      return {
        functionCallingConfig: {
          mode: FunctionCallingConfigMode.ANY,
        },
      }
    default:
      return {
        functionCallingConfig: {
          mode:
            tools && tools?.length > 0
              ? FunctionCallingConfigMode.AUTO
              : FunctionCallingConfigMode.NONE,
        },
      }
  }
}

export const convertTools = (
  tools: OpenAI.Chat.Completions.ChatCompletionTool[] | undefined
): Tool[] | undefined => {
  if (tools === undefined) {
    return undefined
  }

  return tools.map((tool) => {
    let parameters = tool.function.parameters
    if (parameters && typeof parameters === 'object') {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const { required, properties, ...params } = parameters as Schema
      Object.entries(properties).forEach(([key, value]) => {
        if (typeof value === 'object') {
          const { exclusiveMinimum, exclusiveMaximum, format, type, ...rest } =
            value as unknown as {
              exclusiveMinimum?: number
              exclusiveMaximum?: number
              [key: string]: any
            }
          properties[key] = {
            ...(exclusiveMinimum ? { minimum: exclusiveMinimum } : {}),
            ...(exclusiveMaximum ? { maximum: exclusiveMaximum } : {}),
            ...(format === 'enum' ||
            (format === 'date-time' && type === 'string')
              ? { format }
              : {}),
            type,
            ...rest,
          }
        }
      })
      parameters = {
        properties,
        ...params,
      }
    }
    return {
      functionDeclarations: [
        {
          name: tool.function.name,
          description: tool.function.description,
          // We can cast this directly to Google's type because they both use JSON Schema
          // OpenAI just uses a generic Record<string, unknown> type for this.
          parameters,
        },
      ],
    }
  })
}

export const convertResponse = async (
  result: GenerateContentResponse,
  model: string,
  timestamp: number
): Promise<CompletionResponse> => {
  return {
    id: null,
    object: 'chat.completion',
    created: timestamp,
    model,
    choices:
      result.candidates?.map((candidate) => {
        return {
          index: candidate.index,
          finish_reason: candidate.finishReason
            ? convertFinishReason(
                candidate.finishReason,
                candidate.content?.parts
              )
            : 'stop',
          message: convertResponseMessage(candidate),
          // Google does not support logprobs
          logprobs: null,
          // There are also some other fields that Google returns that are not supported in the OpenAI format such as citations and safety ratings
        }
      }) ?? [],
    usage: result.usageMetadata
      ? convertUsageData(result.usageMetadata)
      : undefined,
  }
}

async function* convertStreamResponse(
  result: AsyncGenerator<GenerateContentResponse, any, any>,
  model: string,
  timestamp: number
): StreamCompletionResponse {
  for await (const chunk of result) {
    const text = chunk.text
    yield {
      id: null,
      object: 'chat.completion.chunk',
      created: timestamp,
      model,
      choices:
        chunk.candidates?.map((candidate) => {
          return {
            index: candidate.index,
            finish_reason: candidate.finishReason
              ? convertFinishReason(
                  candidate.finishReason,
                  candidate.content?.parts
                )
              : 'stop',
            delta: {
              content: text,
              tool_calls: convertStreamToolCalls(candidate),
              role: 'assistant',
            },
            logprobs: null,
          }
        }) ?? [],
      usage: chunk.usageMetadata
        ? convertUsageData(chunk.usageMetadata)
        : undefined,
    }
  }
}

// To support a new provider, we just create a handler for them extending the BaseHandler class and implement the create method.
// Then we update the Handlers object in src/handlers/utils.ts to include the new handler.
export class GeminiHandler extends BaseHandler<GeminiModel> {
  async create(
    body: ProviderCompletionParams<'gemini'>,
    options?: RequestOptions
  ): Promise<CompletionResponse | StreamCompletionResponse> {
    this.validateInputs(body)

    const apiKey = this.opts.apiKey ?? process.env.GEMINI_API_KEY
    if (apiKey === undefined) {
      throw new InputError(
        'API key is required for Gemini, define GEMINI_API_KEY in your environment or specifty the apiKey option.'
      )
    }

    const responseMimeType =
      body.response_format?.type === 'json_object'
        ? 'application/json'
        : undefined
    const stop = typeof body.stop === 'string' ? [body.stop] : body.stop

    // Create request options for custom baseURL and headers support
    const requestOptions: HttpOptions = {}
    if (
      this.opts.baseURL &&
      !this.opts.baseURL.includes('https://generativelanguage.googleapis.com')
    ) {
      requestOptions.baseUrl = this.opts.baseURL
    }
    if (this.opts.defaultHeaders) {
      requestOptions.headers = this.opts.defaultHeaders
    }

    const gen = new GoogleGenAI({ apiKey, httpOptions: requestOptions })

    const { contents, systemInstruction } = await convertMessagesToContents(
      body.messages
    )
    const params: GenerateContentParameters = {
      model: body.model,
      contents,
      config: {
        toolConfig: convertToolConfig(body.tool_choice, body.tools),
        tools: convertTools(body.tools),
        systemInstruction,
        maxOutputTokens: body.max_tokens ?? undefined,
        temperature: body.temperature ?? undefined,
        topP: body.top_p ?? undefined,
        stopSequences: stop ?? undefined,
        candidateCount: body.n ?? undefined,
        responseMimeType,
        abortSignal: options?.signal,
      },
    }

    const timestamp = getTimestamp()

    if (body.stream) {
      const result = await gen.models.generateContentStream(params) // options/
      return convertStreamResponse(result, body.model, timestamp)
    } else {
      const result = await gen.models.generateContent(params)
      return convertResponse(result, body.model, timestamp)
    }
  }
}
